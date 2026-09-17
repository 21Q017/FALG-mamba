#!/usr/bin/env python3
"""Evaluate a FALG-Mamba checkpoint on the In-the-Wild audio deepfake set.

Defaults are tailored to the user's layout:
    ~/autodl-tmp/data/release_in_the_wild

The script can auto-discover common protocol/metadata files and accepts both
whitespace-delimited and CSV/TSV formats. Larger model scores mean bonafide.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Keep CPU thread settings sane on multi-worker servers.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils_eval_extra import _read_audio, pad_or_cut
from eval_metrics_DF import compute_eer
from model_scripts.AA_LG_Mamba_MLF_PG import Model

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
    """Return 1 for bonafide, 0 for spoof, None for an unknown token."""
    token = normalize_token(value)
    compact = re.sub(r"[\s_]+", "-", token)
    if token in BONAFIDE_ALIASES or compact in BONAFIDE_ALIASES:
        return 1
    if token in SPOOF_ALIASES or compact in SPOOF_ALIASES:
        return 0
    # Numeric labels are intentionally conservative. A file with an explicit
    # header is handled separately; headerless 0/1 columns need --real-label.
    return None


def discover_protocol(root: Path) -> Path:
    preferred = [
        "protocol.txt",
        "in_the_wild.txt",
        "in-the-wild.txt",
        "in_the_wild.eval.txt",
        "in-the-wild.eval.txt",
        "metadata.csv",
        "meta.csv",
        "labels.csv",
        "protocol.csv",
        "metadata.tsv",
        "protocol.tsv",
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
            f"No protocol/metadata file found under {root}. "
            "Pass it explicitly with --protocol /path/to/file."
        )
    return candidates[0]


def build_audio_index(root: Path) -> Tuple[Dict[str, Path], Dict[str, Path], int]:
    """Index audio by relative path and stem, reporting ambiguous stems."""
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
            f"[audio-index][WARN] {ambiguous} duplicated stems were not indexed by stem; "
            "their protocol entries must include a relative path or full filename.",
            flush=True,
        )
    return by_rel, by_stem, total


def resolve_audio_token(token: object, root: Path, by_rel: Dict[str, Path], by_stem: Dict[str, Path]) -> Optional[Path]:
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
    if Path(normalized).name in by_rel:
        return by_rel[Path(normalized).name]

    stem = Path(normalized).stem
    if stem in by_stem:
        return by_stem[stem]

    for ext in AUDIO_EXTS:
        cand = root / f"{normalized}{ext}"
        if cand.is_file():
            return cand
    return None


def sniff_rows(path: Path) -> Tuple[List[List[str]], Optional[List[str]]]:
    """Read CSV/TSV/whitespace protocol into rows; return optional header."""
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"Protocol is empty: {path}")

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
        rows = [re.split(r"\s+", ln.strip()) for ln in lines]
    else:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(lines, delimiter=delimiter, skipinitialspace=True)
        ]

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


def load_protocol(
    protocol: Path,
    root: Path,
    by_rel: Dict[str, Path],
    by_stem: Dict[str, Path],
    real_label: int,
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
        if label_idx is not None:
            numeric_label_mode = True

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
                        # --real-label tells which numeric value denotes bonafide.
                        label = 1 if value == real_label else 0
                elif header and header[label_idx] == "is_fake" and token in {"true", "false"}:
                    label = 0 if token == "true" else 1
                elif header and header[label_idx] == "is_real" and token in {"true", "false"}:
                    label = 1 if token == "true" else 0

        if label is None:
            for cell in reversed(row):
                label = parse_label(cell)
                if label is not None:
                    break

        # Headerless two-column/list protocols sometimes use numeric 0/1 labels.
        # Accept this only when exactly one binary token is present, avoiding a
        # blind assumption about arbitrary numeric metadata columns.
        if label is None:
            binary_cells = [normalize_token(cell) for cell in row if normalize_token(cell) in {"0", "1"}]
            if len(binary_cells) == 1:
                value = int(binary_cells[0])
                label = 1 if value == real_label else 0

        audio_path: Optional[Path] = None
        if path_idx is not None and path_idx < len(row):
            audio_path = resolve_audio_token(row[path_idx], root, by_rel, by_stem)
        if audio_path is None:
            # Prefer tokens that look like audio paths, then try every cell.
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
                raise ValueError(
                    f"Conflicting labels for {key}: existing={labels[key]} new={label} at line {line_no}"
                )
            continue
        keys.append(key)
        paths.append(audio_path)
        labels[key] = label

    if duplicates:
        print(f"[protocol][WARN] ignored {duplicates} duplicate entries", flush=True)
    if not keys:
        preview = "\n".join(f"  line {n}: {row}" for n, row in rejected[:5])
        raise ValueError(
            f"Could not parse any labeled audio trials from {protocol}.\n"
            f"Rejected examples:\n{preview}\n"
            "Pass --protocol explicitly and, for numeric labels, set --real-label 0 or 1."
        )
    return keys, paths, labels, rejected


class ITWDataset(Dataset):
    def __init__(self, keys: Sequence[str], paths: Sequence[Path], target_len: int, on_error: str):
        self.keys = list(keys)
        self.paths = list(paths)
        self.target_len = int(target_len)
        self.on_error = on_error
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be skip or raise")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        path = self.paths[idx]
        try:
            wav = pad_or_cut(_read_audio(path), self.target_len, random_crop=False)
            return torch.from_numpy(wav), key, True
        except Exception as exc:
            print(
                f"[ITW][audio-error] idx={idx} key={key} path={path} "
                f"type={type(exc).__name__} error={exc}",
                file=sys.stderr,
                flush=True,
            )
            if self.on_error == "raise":
                raise
            return torch.zeros(self.target_len, dtype=torch.float32), key, False


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
    elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: value for k, value in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"[checkpoint] loaded strictly: {path}", flush=True)


def build_model(args: argparse.Namespace, device: torch.device) -> Model:
    model_args = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "n_layer": args.n_layer,
        "num_classes": 2,
        "n_query": args.n_query,
        "scan": args.scan,
        "use_oc": True,
        "logit_source": "cls" if args.score_mode == "raw" else "oc",
        "oc_alpha": args.oc_alpha,
        "oc_m_real": args.oc_m_real,
        "oc_m_fake": args.oc_m_fake,
        "mlf_fusion": args.mlf_fusion,
        "mlf_prior_init": args.mlf_prior_init,
        "mlf_prior_strength": args.mlf_prior_strength,
        "mlf_align_level": args.mlf_align_level,
        "sinc_channels": args.sinc_channels,
        "freq_pool": args.freq_pool,
        "pool_heads": args.pool_heads,
        "headdim": args.headdim,
        "dropout": args.dropout,
        "use_gate": True,
        "freq_flatten": args.freq_flatten,
        "flatten_mode": args.flatten_mode,
        "preact_fix": args.preact_fix,
        # Exp B switch: fixed_band adds no learnable params; k_query default unchanged.
        "freq_tokenizer": getattr(args, "freq_tokenizer", "k_query"),
    }
    model = Model(model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] params={n_params/1e6:.3f}M score_mode={args.score_mode} "
        f"align={args.mlf_align_level} fusion={args.mlf_fusion} device={device}",
        flush=True,
    )
    return model


def autocast_context(args: argparse.Namespace, device: torch.device):
    enabled = bool(args.amp and device.type == "cuda")
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: Dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
    model.eval()
    keys_all: List[str] = []
    labels_all: List[int] = []
    scores_all: List[float] = []
    unreadable: List[str] = []

    with torch.inference_mode():
        for data, keys, valid in tqdm(loader, desc="ITW evaluation", ncols=110):
            valid_mask = torch.as_tensor(valid, dtype=torch.bool)
            if not bool(valid_mask.any()):
                unreadable.extend(str(k) for k in keys)
                continue
            if not bool(valid_mask.all()):
                unreadable.extend(str(keys[i]) for i in range(len(keys)) if not bool(valid_mask[i]))
                data = data[valid_mask]
                keys = [str(keys[i]) for i in range(len(keys)) if bool(valid_mask[i])]
            else:
                keys = [str(k) for k in keys]

            data = data.to(device, non_blocking=True)
            with autocast_context(args, device):
                output = model(data)
                logits = output[0] if isinstance(output, tuple) else output
            score = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
            keys_all.extend(keys)
            labels_all.extend(labels[k] for k in keys)
            scores_all.extend(float(x) for x in score)

    return keys_all, np.asarray(labels_all, dtype=np.int64), np.asarray(scores_all, dtype=np.float64), unreadable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load a checkpoint and evaluate In-the-Wild EER")
    p.add_argument("--ckpt", required=True, help="best_df_model.pt / best_joint_model.pt / averaged checkpoint")
    p.add_argument("--itw_root", default="~/autodl-tmp/data/release_in_the_wild")
    p.add_argument("--protocol", default="", help="optional protocol path; auto-discovered when omitted")
    p.add_argument("--out_dir", default="./eval_itw")
    p.add_argument("--tag", default="itw")
    p.add_argument("--real-label", type=int, default=1, choices=[0, 1],
                   help="for headered numeric-label protocols: which value means bonafide")
    p.add_argument("--eval_len", type=int, default=64600)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--audio_error", choices=["skip", "raise"], default="skip")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no_tf32", dest="tf32", action="store_false")

    # Architecture flags must match the checkpoint. Defaults match the current best configuration.
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--d_state", type=int, default=16)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_query", type=int, default=4)
    p.add_argument("--scan", choices=["alt", "uni", "bi"], default="alt")
    p.add_argument("--sinc_channels", type=int, default=70)
    p.add_argument("--pool_heads", type=int, default=4)
    p.add_argument("--headdim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freq_pool", type=int, default=3)
    p.add_argument("--mlf_align_level", choices=["shallow", "middle", "deep"], default="middle")
    p.add_argument("--mlf_fusion", choices=["attn", "prior_attn", "prior_only", "mean"], default="attn")
    p.add_argument("--mlf_prior_init", default="0.33,0.33,0.34")
    p.add_argument("--mlf_prior_strength", type=float, default=0.0)
    p.add_argument("--freq_flatten", action="store_true", default=False)
    p.add_argument("--flatten_mode", choices=["avg", "max"], default="avg")
    p.add_argument("--freq_tokenizer", choices=["k_query", "fixed_band"], default="k_query",
                   help="Exp B: k_query (default) | fixed_band (equal-width sub-bands, param-free)")
    p.add_argument("--preact_fix", dest="preact_fix", action="store_true", default=True)
    p.add_argument("--no_preact_fix", dest="preact_fix", action="store_false")
    p.add_argument("--score_mode", choices=["raw", "oc"], default="raw")
    p.add_argument("--oc_alpha", type=float, default=20.0)
    p.add_argument("--oc_m_real", type=float, default=0.9)
    p.add_argument("--oc_m_fake", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.itw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--itw_root is not a directory: {root}")
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {ckpt}")
    protocol = Path(args.protocol).expanduser().resolve() if args.protocol else discover_protocol(root)
    if not protocol.is_file():
        raise FileNotFoundError(f"--protocol not found: {protocol}")

    print(f"[ITW] root={root}", flush=True)
    print(f"[ITW] protocol={protocol}", flush=True)
    by_rel, by_stem, audio_total = build_audio_index(root)
    print(f"[audio-index] files={audio_total} unique_stems={len(by_stem)}", flush=True)
    keys, paths, labels, rejected = load_protocol(
        protocol, root, by_rel, by_stem, real_label=args.real_label
    )
    counts = Counter(labels.values())
    print(
        f"[protocol] accepted={len(keys):,} bonafide={counts[1]:,} spoof={counts[0]:,} "
        f"rejected={len(rejected):,}",
        flush=True,
    )
    if rejected:
        print("[protocol][WARN] first rejected rows:", flush=True)
        for line_no, row in rejected[:5]:
            print(f"  line {line_no}: {row}", flush=True)

    if counts[1] == 0 or counts[0] == 0:
        raise ValueError(
            f"Both classes are required for EER, got bonafide={counts[1]} spoof={counts[0]}. "
            "Check --protocol and --real-label."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = ITWDataset(keys, paths, target_len=args.eval_len, on_error=args.audio_error)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_model(args, device)
    load_checkpoint(model, ckpt, device)
    keys_scored, y, scores, unreadable = evaluate(model, loader, labels, device, args)
    if y.size == 0:
        raise RuntimeError("No readable trials were scored")

    bona = scores[y == 1]
    spoof = scores[y == 0]
    eer_frac, threshold = compute_eer(bona, spoof)
    inv_frac, inv_threshold = compute_eer(-bona, -spoof)
    eer = float(eer_frac * 100.0)
    inv_eer = float(inv_frac * 100.0)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / f"{args.tag}_scores.txt"
    detail_path = out_dir / f"{args.tag}_scores_with_labels.tsv"
    summary_path = out_dir / f"{args.tag}_summary.json"
    unreadable_path = out_dir / f"{args.tag}_unreadable.txt"

    with score_path.open("w", encoding="utf-8") as f:
        for key, score in zip(keys_scored, scores):
            f.write(f"{key} {score:.10f}\n")
    with detail_path.open("w", encoding="utf-8") as f:
        f.write("utt\tlabel\tscore\n")
        for key, label, score in zip(keys_scored, y, scores):
            f.write(f"{key}\t{'bonafide' if label == 1 else 'spoof'}\t{score:.10f}\n")
    if unreadable:
        unreadable_path.write_text("\n".join(sorted(set(unreadable))) + "\n", encoding="utf-8")

    summary = {
        "dataset": "In-the-Wild",
        "checkpoint": str(ckpt),
        "itw_root": str(root),
        "protocol": str(protocol),
        "score_convention": "larger score means bonafide; score=logit[1]-logit[0]",
        "score_mode": args.score_mode,
        "n_protocol": len(keys),
        "n_scored": int(y.size),
        "n_bonafide": int((y == 1).sum()),
        "n_spoof": int((y == 0).sum()),
        "n_unreadable": len(set(unreadable)),
        "n_protocol_rows_rejected": len(rejected),
        "eer_percent": eer,
        "eer_threshold": float(threshold),
        "inverted_eer_percent_diagnostic": inv_eer,
        "inverted_threshold_diagnostic": float(inv_threshold),
        "mean_bonafide_score": float(bona.mean()),
        "mean_spoof_score": float(spoof.mean()),
        "score_separation": float(bona.mean() - spoof.mean()),
        "architecture": {
            "d_model": args.d_model,
            "d_state": args.d_state,
            "n_layer": args.n_layer,
            "n_query": args.n_query,
            "scan": args.scan,
            "mlf_align_level": args.mlf_align_level,
            "mlf_fusion": args.mlf_fusion,
            "preact_fix": args.preact_fix,
            "freq_flatten": args.freq_flatten,
            "freq_tokenizer": getattr(args, "freq_tokenizer", "k_query"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=" * 78)
    print(f"IN-THE-WILD RESULT   tag={args.tag}")
    print("=" * 78)
    print(
        f"  ITW  N={y.size:,}  bona={int((y == 1).sum()):,}  spoof={int((y == 0).sum()):,}  "
        f"EER={eer:.2f}%  threshold={float(threshold):.6f}"
    )
    print(
        f"  mean_score: bona={bona.mean():.6f}  spoof={spoof.mean():.6f}  "
        f"separation={bona.mean() - spoof.mean():.6f}"
    )
    print("=" * 78)
    print(f"scores : {score_path}")
    print(f"details: {detail_path}")
    print(f"summary: {summary_path}")
    if unreadable:
        print(f"unreadable ({len(set(unreadable))}): {unreadable_path}")
    if inv_eer + 1e-6 < eer:
        print(
            f"[WARN] Inverted-score EER is lower ({inv_eer:.2f}% vs {eer:.2f}%). "
            "Do not silently flip scores; first verify protocol labels and --real-label.",
            flush=True,
        )


if __name__ == "__main__":
    main()
