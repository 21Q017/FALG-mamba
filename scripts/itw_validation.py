"""In-the-Wild protocol parsing and deterministic validation dataset.

This module is shared by the training-time ITW validator.  Labels follow the
codebase convention: spoof=0, bonafide=1.  A dataset item contains a fixed
number of deterministic crops so DataLoader collation remains simple.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils_eval_extra import _read_audio, pad_or_cut

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
BONAFIDE_ALIASES = {
    "bonafide", "bona-fide", "bona_fide", "real", "genuine", "human",
    "natural", "authentic", "true", "0_real", "1_real",
}
SPOOF_ALIASES = {
    "spoof", "fake", "deepfake", "synthetic", "generated", "tts", "vc",
    "converted", "0_fake", "1_fake",
}


def normalize_token(value: object) -> str:
    return str(value).strip().strip("\"'").lower()


def parse_label(value: object) -> Optional[int]:
    token = normalize_token(value)
    compact = re.sub(r"[\s_]+", "-", token)
    if token in BONAFIDE_ALIASES or compact in BONAFIDE_ALIASES:
        return 1
    if token in SPOOF_ALIASES or compact in SPOOF_ALIASES:
        return 0
    return None


def discover_protocol(root: Path) -> Path:
    preferred = [
        "protocol.txt", "in_the_wild.txt", "in-the-wild.txt",
        "in_the_wild.eval.txt", "in-the-wild.eval.txt", "metadata.csv",
        "meta.csv", "labels.csv", "protocol.csv", "metadata.tsv", "protocol.tsv",
    ]
    for name in preferred:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    candidates: List[Path] = []
    for pattern in ("*.txt", "*.csv", "*.tsv", "*.lst"):
        for path in root.rglob(pattern):
            low = path.name.lower()
            if any(key in low for key in ("protocol", "meta", "label", "wild", "trial")):
                candidates.append(path)
    candidates = sorted(set(candidates), key=lambda p: (len(p.parts), len(p.name), str(p)))
    if not candidates:
        raise FileNotFoundError(
            f"No ITW protocol/metadata file found under {root}. "
            "Pass it explicitly with --itw_protocol /path/to/file."
        )
    return candidates[0]


def build_audio_index(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path], int]:
    by_rel: Dict[str, Path] = {}
    stem_candidates: Dict[str, List[Path]] = {}
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        total += 1
        rel = path.relative_to(root).as_posix()
        by_rel[rel] = path
        by_rel[Path(rel).name] = path
        stem_candidates.setdefault(path.stem, []).append(path)
    by_stem: Dict[str, Path] = {}
    ambiguous = 0
    for stem, paths in stem_candidates.items():
        if len(paths) == 1:
            by_stem[stem] = paths[0]
        else:
            ambiguous += 1
    if ambiguous:
        print(
            f"[ITW][audio-index][WARN] {ambiguous} duplicated stems are not indexed by stem; "
            "their protocol entries must contain a relative path or filename.",
            flush=True,
        )
    return by_rel, by_stem, total


def resolve_audio_token(
    token: object,
    root: Path,
    by_rel: Dict[str, Path],
    by_stem: Dict[str, Path],
) -> Optional[Path]:
    raw = str(token).strip().strip("\"'")
    if not raw:
        return None
    raw = raw.replace("\\", "/")
    p = Path(raw).expanduser()
    if p.is_absolute() and p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return p
    normalized = raw.lstrip("./")
    if normalized in by_rel:
        return by_rel[normalized]
    name = Path(normalized).name
    if name in by_rel:
        return by_rel[name]
    stem = Path(normalized).stem
    if stem in by_stem:
        return by_stem[stem]
    for ext in AUDIO_EXTS:
        cand = root / f"{normalized}{ext}"
        if cand.is_file():
            return cand
    return None


def sniff_rows(path: Path) -> Tuple[List[List[str]], Optional[List[str]]]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"ITW protocol is empty: {path}")
    suffix = path.suffix.lower()
    delimiter: Optional[str]
    if suffix == ".csv":
        delimiter = ","
    elif suffix == ".tsv":
        delimiter = "\t"
    else:
        sample = "\n".join(lines[:20])
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
        except csv.Error:
            delimiter = None
    if delimiter is None:
        rows = [re.split(r"\s+", line.strip()) for line in lines]
    else:
        rows = [[cell.strip() for cell in row] for row in csv.reader(lines, delimiter=delimiter, skipinitialspace=True)]
    header = None
    if rows:
        first = [normalize_token(x) for x in rows[0]]
        header_words = {
            "file", "filename", "file_name", "path", "audio", "audio_path",
            "utt", "utt_id", "utterance", "id", "label", "class", "target",
            "is_fake", "is_real", "type",
        }
        if any(x in header_words for x in first):
            header = first
            rows = rows[1:]
    return rows, header


def _header_index(header: Sequence[str], names: Iterable[str]) -> Optional[int]:
    lowered = [normalize_token(x) for x in header]
    for name in names:
        if name in lowered:
            return lowered.index(name)
    return None


def load_itw_protocol(
    protocol: Path,
    root: Path,
    by_rel: Dict[str, Path],
    by_stem: Dict[str, Path],
    real_label: int = 1,
) -> Tuple[List[str], List[Path], Dict[str, int], List[Tuple[int, List[str]]]]:
    rows, header = sniff_rows(protocol)
    path_idx = label_idx = None
    numeric_label_mode = False
    if header is not None:
        path_idx = _header_index(
            header,
            ("path", "audio_path", "file", "filename", "file_name", "audio", "wav", "utt", "utt_id", "utterance", "id"),
        )
        label_idx = _header_index(header, ("label", "class", "target", "type", "is_fake", "is_real"))
        numeric_label_mode = label_idx is not None

    keys: List[str] = []
    paths: List[Path] = []
    labels: Dict[str, int] = {}
    rejected: List[Tuple[int, List[str]]] = []
    duplicates = 0

    for line_no, row in enumerate(rows, start=2 if header else 1):
        if not row:
            continue
        label: Optional[int] = None
        if label_idx is not None and label_idx < len(row):
            label = parse_label(row[label_idx])
            if label is None and numeric_label_mode:
                token = normalize_token(row[label_idx])
                if token in {"0", "1"}:
                    value = int(token)
                    column_name = header[label_idx] if header else ""
                    if column_name == "is_fake":
                        label = 0 if value == 1 else 1
                    elif column_name == "is_real":
                        label = 1 if value == 1 else 0
                    else:
                        label = 1 if value == int(real_label) else 0
                elif header and header[label_idx] == "is_fake" and token in {"true", "false"}:
                    label = 0 if token == "true" else 1
                elif header and header[label_idx] == "is_real" and token in {"true", "false"}:
                    label = 1 if token == "true" else 0
        if label is None:
            for cell in reversed(row):
                label = parse_label(cell)
                if label is not None:
                    break
        if label is None:
            binary = [normalize_token(cell) for cell in row if normalize_token(cell) in {"0", "1"}]
            if len(binary) == 1:
                value = int(binary[0])
                label = 1 if value == int(real_label) else 0

        audio_path: Optional[Path] = None
        if path_idx is not None and path_idx < len(row):
            audio_path = resolve_audio_token(row[path_idx], root, by_rel, by_stem)
        if audio_path is None:
            ordered = sorted(
                row,
                key=lambda x: 0 if Path(str(x).strip().strip("\"'")).suffix.lower() in AUDIO_EXTS else 1,
            )
            for cell in ordered:
                audio_path = resolve_audio_token(cell, root, by_rel, by_stem)
                if audio_path is not None:
                    break
        if label is None or audio_path is None:
            rejected.append((line_no, row))
            continue

        rel = audio_path.relative_to(root).as_posix() if root in audio_path.parents else audio_path.name
        key = rel
        if key in labels:
            duplicates += 1
            if labels[key] != label:
                raise ValueError(f"Conflicting labels for ITW trial {key} at protocol line {line_no}")
            continue
        keys.append(key)
        paths.append(audio_path)
        labels[key] = label

    if duplicates:
        print(f"[ITW][protocol][WARN] ignored {duplicates} duplicate entries", flush=True)
    if not keys:
        preview = "\n".join(f"  line {n}: {row}" for n, row in rejected[:5])
        raise ValueError(
            f"Could not parse labeled ITW trials from {protocol}.\nRejected examples:\n{preview}\n"
            "Pass --itw_protocol explicitly and set --itw_real_label for numeric labels."
        )
    return keys, paths, labels, rejected


def stratified_sample_trials(
    keys: Sequence[str],
    paths: Sequence[Path],
    labels: Dict[str, int],
    ratio: float,
    seed: int,
) -> Tuple[List[str], List[Path], Dict[str, int]]:
    ratio = float(ratio)
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"--itw_sample_ratio must be in (0, 1], got {ratio}")
    if ratio >= 1.0:
        return list(keys), list(paths), dict(labels)
    rng = np.random.RandomState(int(seed))
    path_by_key = dict(zip(keys, paths))
    selected: List[str] = []
    for label in (1, 0):
        group = [key for key in keys if labels[key] == label]
        if not group:
            continue
        n = max(1, int(round(len(group) * ratio)))
        idx = rng.choice(len(group), size=min(n, len(group)), replace=False)
        selected.extend(group[i] for i in sorted(idx))
    rng.shuffle(selected)
    return selected, [path_by_key[key] for key in selected], {key: labels[key] for key in selected}


def deterministic_crops(wav: np.ndarray, target_len: int, num_crops: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    num_crops = max(1, int(num_crops))
    if wav.size <= target_len:
        crop = pad_or_cut(wav, target_len=target_len, random_crop=False)
        return np.repeat(crop[None, :], num_crops, axis=0)
    max_start = wav.size - target_len
    starts = np.linspace(0, max_start, num=num_crops, dtype=np.int64)
    return np.stack([wav[int(start):int(start) + target_len] for start in starts], axis=0).astype(np.float32, copy=False)


class ITWValidationDataset(Dataset):
    def __init__(
        self,
        keys: Sequence[str],
        paths: Sequence[Path],
        target_len: int = 64600,
        num_crops: int = 1,
        on_error: str = "skip",
    ):
        self.keys = list(keys)
        self.paths = [Path(path) for path in paths]
        self.target_len = int(target_len)
        self.num_crops = max(1, int(num_crops))
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be skip or raise")
        self.on_error = on_error

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        path = self.paths[idx]
        try:
            crops = deterministic_crops(_read_audio(path), self.target_len, self.num_crops)
            return torch.from_numpy(crops), key, True
        except Exception as exc:
            print(
                f"[ITW][audio-error] idx={idx} key={key} path={path} "
                f"type={type(exc).__name__} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            if self.on_error == "raise":
                raise
            return torch.zeros((self.num_crops, self.target_len), dtype=torch.float32), key, False


def build_itw_validation_data(args):
    root = Path(args.itw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--itw_root is not a directory: {root}")
    protocol = Path(args.itw_protocol).expanduser().resolve() if args.itw_protocol else discover_protocol(root)
    if not protocol.is_file():
        raise FileNotFoundError(f"--itw_protocol not found: {protocol}")
    by_rel, by_stem, audio_total = build_audio_index(root)
    keys, paths, labels, rejected = load_itw_protocol(
        protocol, root, by_rel, by_stem, real_label=args.itw_real_label
    )
    full_counts = Counter(labels.values())
    keys, paths, labels = stratified_sample_trials(
        keys, paths, labels, ratio=args.itw_sample_ratio, seed=args.itw_sampling_seed
    )
    counts = Counter(labels.values())
    print(f"[ITW] root={root}", flush=True)
    print(f"[ITW] protocol={protocol}", flush=True)
    print(
        f"[ITW] audio_files={audio_total:,} protocol_full={sum(full_counts.values()):,} "
        f"full_bona={full_counts[1]:,} full_spoof={full_counts[0]:,} rejected={len(rejected):,}",
        flush=True,
    )
    print(
        f"[ITW] validation N={len(keys):,} bona={counts[1]:,} spoof={counts[0]:,} "
        f"sample_ratio={args.itw_sample_ratio:g} seed={args.itw_sampling_seed} crops={args.itw_num_crops}",
        flush=True,
    )
    if counts[1] == 0 or counts[0] == 0:
        raise ValueError(f"ITW EER needs both classes, got bona={counts[1]} spoof={counts[0]}")
    if rejected:
        print("[ITW][protocol][WARN] first rejected rows:", flush=True)
        for line_no, row in rejected[:5]:
            print(f"  line {line_no}: {row}", flush=True)
    dataset = ITWValidationDataset(
        keys, paths, target_len=args.eval_len, num_crops=args.itw_num_crops, on_error=args.audio_error
    )
    return dataset, labels, protocol
