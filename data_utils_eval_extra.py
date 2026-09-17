"""Generic waveform evaluation dataset utilities with explicit load diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_AUDIO_EXTS = {'.flac', '.wav', '.mp3'}


def _stderr(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _candidate_audio_paths(audio_dir: Path, utt_id: str, ext: str = '.flac') -> List[Path]:
    p = Path(str(utt_id))
    if p.suffix:
        return [audio_dir / p.name, audio_dir / p]
    return [
        audio_dir / f"{utt_id}{ext}",
        audio_dir / f"{utt_id}.flac",
        audio_dir / f"{utt_id}.wav",
    ]


def _quick_resolve_audio_path(audio_dir: Path, utt_id: str, ext: str = '.flac') -> Optional[Path]:
    for cand in _candidate_audio_paths(audio_dir, utt_id, ext):
        if cand.exists():
            return cand
    return None


def summarize_audio_dataset(keys: Iterable[str], audio_dir, ext: str = '.flac', name: str = 'dataset', max_missing: int = 10, max_examples: int = 3):
    """Print a cheap direct-path diagnostic for a dataset.

    This intentionally avoids recursive search for speed on large ASVspoof eval
    sets. Actual item loading still uses _resolve_audio_path and can resolve
    nested layouts.
    """
    keys = list(keys)
    audio_dir = Path(audio_dir)
    print(f"[DatasetLoad][{name}] keys={len(keys)} audio_dir={audio_dir} exists={audio_dir.is_dir()} ext={ext}", flush=True)
    if not audio_dir.is_dir():
        print(f"[DatasetLoad][{name}][ERROR] audio_dir is not a directory: {audio_dir}", flush=True)
        return

    missing = []
    examples = []
    for utt in keys:
        direct = _quick_resolve_audio_path(audio_dir, utt, ext)
        if direct is None:
            if len(missing) < max_missing:
                cand_str = ' | '.join(str(p) for p in _candidate_audio_paths(audio_dir, utt, ext))
                missing.append((utt, cand_str))
        elif len(examples) < max_examples:
            examples.append((utt, str(direct)))

    direct_found = len(keys) - len(missing) if len(missing) < max_missing else None
    if examples:
        print(f"[DatasetLoad][{name}] example resolved files:", flush=True)
        for i, (utt, path) in enumerate(examples, 1):
            print(f"  [{i}] utt={utt} path={path}", flush=True)
    if missing:
        print(f"[DatasetLoad][{name}][WARN] missing direct-path examples shown={len(missing)}. Recursive lookup may still work for nested layouts.", flush=True)
        for i, (utt, cand_str) in enumerate(missing, 1):
            print(f"  [missing #{i}] utt={utt} candidates={cand_str}", flush=True)
    else:
        print(f"[DatasetLoad][{name}] direct-path check passed for all {len(keys)} keys", flush=True)


def _read_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"audio path is not a file: {path}")

    sf_exc = None
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            except Exception as exc:
                raise RuntimeError(f"librosa failed to resample {path} from {sr} to {target_sr}: {repr(exc)}") from exc
        return wav.astype(np.float32, copy=False)
    except Exception as exc:
        sf_exc = exc

    try:
        import torchaudio
        wav_t, sr = torchaudio.load(str(path))
        if wav_t.size(0) > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        return wav_t.squeeze(0).numpy().astype(np.float32, copy=False)
    except Exception as ta_exc:
        raise RuntimeError(
            f"Failed to read audio: {path}\n"
            f"  soundfile error: {repr(sf_exc)}\n"
            f"  torchaudio error: {repr(ta_exc)}"
        ) from ta_exc


def pad_or_cut(wav: np.ndarray, target_len: int = 64600, random_crop: bool = False) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if len(wav) >= target_len:
        if random_crop and len(wav) > target_len:
            start = np.random.randint(0, len(wav) - target_len + 1)
        else:
            start = 0
        return wav[start:start + target_len].astype(np.float32, copy=False)
    reps = int(np.ceil(target_len / max(len(wav), 1)))
    if len(wav) == 0:
        wav = np.zeros(target_len, dtype=np.float32)
    else:
        wav = np.tile(wav, reps)[:target_len]
    return wav.astype(np.float32, copy=False)


def _resolve_audio_path(audio_dir: Path, utt_id: str, ext: str = '.flac') -> Path:
    audio_dir = Path(audio_dir)
    p = Path(str(utt_id))
    candidates = _candidate_audio_paths(audio_dir, str(utt_id), ext)
    for cand in candidates:
        if cand.exists():
            return cand
    # Last resort: recursive search by stem. This is slower but useful for ITW layouts.
    stem = p.stem if p.suffix else str(utt_id)
    if audio_dir.is_dir():
        for cand in audio_dir.rglob(f"{stem}.*"):
            if cand.suffix.lower() in _AUDIO_EXTS:
                return cand
    cand_str = ' | '.join(str(c) for c in candidates)
    raise FileNotFoundError(f"Cannot find audio for utt_id={utt_id} under {audio_dir}; tried: {cand_str}")


BAD_AUDIO_SENTINEL = '__UNREADABLE__'


class Dataset_Audio_eval(Dataset):
    def __init__(self, keys: Iterable[str], audio_dir: str, ext: str = '.flac', target_len: int = 64600, name: str = 'AudioEval', verbose: bool = True, on_error: str = 'skip'):
        self.keys = list(keys)
        self.audio_dir = Path(audio_dir)
        self.ext = ext
        self.target_len = int(target_len)
        self.name = name
        if on_error not in ('skip', 'raise'):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
        self.on_error = on_error
        if verbose:
            summarize_audio_dataset(self.keys, self.audio_dir, ext=self.ext, name=self.name)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        utt = self.keys[idx]
        path = None
        try:
            path = _resolve_audio_path(self.audio_dir, utt, self.ext)
            wav = pad_or_cut(_read_audio(path), self.target_len, random_crop=False)
            return torch.from_numpy(wav), utt
        except Exception as exc:
            msg = (
                f"[DatasetLoad][{self.name}][ERROR] failed to load audio\n"
                f"  idx={idx}\n"
                f"  utt_id={utt}\n"
                f"  audio_dir={self.audio_dir}\n"
                f"  resolved_path={path}\n"
                f"  error_type={type(exc).__name__}\n"
                f"  error={exc}"
            )
            _stderr(msg)
            if self.on_error == 'raise':
                raise
            zeros = np.zeros(self.target_len, dtype=np.float32)
            return torch.from_numpy(zeros), f"{BAD_AUDIO_SENTINEL}{utt}"
