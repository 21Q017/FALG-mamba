"""Raw-waveform datasets for ASVspoof2019 LA training and ASVspoof2021 DF eval.

RawBoost augmentation functions are copied from the uploaded RawBMamba package
and reused here for waveform augmentation.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils_eval_extra import _read_audio, pad_or_cut, _resolve_audio_path, summarize_audio_dataset

try:
    from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise
except Exception:
    LnL_convolutive_noise = ISD_additive_noise = SSI_additive_noise = None


def _rawboost(x: np.ndarray, algo: int, fs: int = 16000) -> np.ndarray:
    """RawBoost augmentation, following the numbering of Tak et al. (ICASSP 2022).

    0 : no augmentation
    1 : LnL convolutive noise
    2 : ISD impulsive signal-dependent noise
    3 : SSI stationary signal-independent noise
    4 : series 1+2+3
    5 : series 1+2
    6 : series 1+3
    7 : series 2+3
    8 : parallel 1+2 (outputs summed, then peak-normalised)

    Parameters are the official RawBoost defaults. NOTE: an earlier version of this
    file used a different (non-standard) numbering with no algo 4 and used
    maxBiasLinNonLin=5 instead of the official 20; results obtained with it are not
    directly comparable to the RawBoost paper.
    """
    if algo <= 0 or LnL_convolutive_noise is None:
        return np.asarray(x, dtype=np.float32)

    def lnl(y):
        # N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
        # minCoeff=10, maxCoeff=100, minG=0, maxG=0, minBiasLinNonLin=5, maxBiasLinNonLin=20
        return LnL_convolutive_noise(y, 5, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, 5, 20, fs)

    def isd(y):
        # P=10, g_sd=2
        return ISD_additive_noise(y, 10, 2)

    def ssi(y):
        # SNRmin=10, SNRmax=40, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
        # minCoeff=10, maxCoeff=100, minG=0, maxG=0
        return SSI_additive_noise(y, 10, 40, 5, 20, 8000, 100, 1000, 10, 100, 0, 0, fs)

    y = x.astype(np.float64, copy=False)
    algo = int(algo)
    if algo == 1:
        y = lnl(y)
    elif algo == 2:
        y = isd(y)
    elif algo == 3:
        y = ssi(y)
    elif algo == 4:
        y = ssi(isd(lnl(y)))
    elif algo == 5:
        y = isd(lnl(y))
    elif algo == 6:
        y = ssi(lnl(y))
    elif algo == 7:
        y = ssi(isd(y))
    elif algo == 8:
        y = _normwav(lnl(y) + isd(y))
    else:
        raise ValueError(f"unknown RawBoost algo={algo} (expected 0-8)")
    return np.asarray(y, dtype=np.float32)


def _normwav(y: np.ndarray) -> np.ndarray:
    peak = np.amax(np.abs(y))
    return y / peak if peak > 0 else y


BAD_AUDIO_SENTINEL = '__UNREADABLE__'


class Dataset_2019LA_train(Dataset):
    def __init__(self, list_IDs: Iterable[str], labels: Dict[str, int], base_dir, target_len: int = 64600, rawboost_algos=None, name: str = '2019LA-train', verbose: bool = True, on_error: str = 'skip'):
        self.list_IDs = list(list_IDs)
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.target_len = int(target_len)
        self.rawboost_algos = list(rawboost_algos or [])
        self.name = name
        if on_error not in ('skip', 'raise'):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
        self.on_error = on_error
        # Accept either ASVspoof2019_LA_train or ASVspoof2019_LA_train/flac.
        self.audio_dir = self.base_dir / 'flac' if (self.base_dir / 'flac').is_dir() else self.base_dir
        if verbose:
            print(f"[DatasetLoad][{self.name}] labels={len(self.labels)} rawboost_algos={self.rawboost_algos}", flush=True)
            summarize_audio_dataset(self.list_IDs, self.audio_dir, ext='.flac', name=self.name)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx, _depth: int = 0):
        utt = self.list_IDs[idx]
        path = None
        try:
            path = _resolve_audio_path(self.audio_dir, utt, '.flac')
            wav = _read_audio(path)
            if self.rawboost_algos:
                algo = random.choice(self.rawboost_algos)
                try:
                    wav = _rawboost(wav, int(algo))
                except Exception as exc:
                    print(
                        f"[DatasetLoad][{self.name}][ERROR] RawBoost failed idx={idx} utt_id={utt} path={path} algo={algo} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
            wav = pad_or_cut(wav, self.target_len, random_crop=True)
            return torch.from_numpy(wav), torch.tensor(int(self.labels[utt]), dtype=torch.long)
        except Exception as exc:
            print(
                f"[DatasetLoad][{self.name}][ERROR] failed to load training audio idx={idx} utt_id={utt} audio_dir={self.audio_dir} resolved_path={path} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if self.on_error == 'raise' or _depth >= 8:
                raise
            # A single corrupt file must not kill a multi-hour run: draw another
            # sample instead. The utterance is logged above so it can be fixed.
            alt = random.randrange(len(self.list_IDs))
            return self.__getitem__(alt, _depth=_depth + 1)


class Dataset_2021DF_eval(Dataset):
    def __init__(self, keys: Iterable[str], audio_dir, target_len: int = 64600, name: str = '2021DF-eval', verbose: bool = True, on_error: str = 'skip'):
        self.keys = list(keys)
        self.audio_dir = Path(audio_dir)
        self.target_len = int(target_len)
        self.name = name
        if on_error not in ('skip', 'raise'):
            raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
        self.on_error = on_error
        if verbose:
            summarize_audio_dataset(self.keys, self.audio_dir, ext='.flac', name=self.name)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        utt = self.keys[idx]
        path = None
        try:
            path = _resolve_audio_path(self.audio_dir, utt, '.flac')
            wav = pad_or_cut(_read_audio(path), self.target_len, random_crop=False)
            return torch.from_numpy(wav), utt
        except Exception as exc:
            print(
                f"[DatasetLoad][{self.name}][ERROR] failed to load eval audio idx={idx} utt_id={utt} audio_dir={self.audio_dir} resolved_path={path} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if self.on_error == 'raise':
                raise
            # Return a zero waveform tagged with a sentinel key. evaluate()/produce_scores()
            # drop these trials from the metric instead of scoring silence, which would
            # quietly corrupt the EER.
            zeros = np.zeros(self.target_len, dtype=np.float32)
            return torch.from_numpy(zeros), f"{BAD_AUDIO_SENTINEL}{utt}"


_ASV_UTT_RE = re.compile(r'^(?:LA|DF)_[A-Z]_[0-9]+$', re.IGNORECASE)

def _pick_utt_id(parts: List[str]) -> str:
    """Pick the true utterance id from ASVspoof metadata rows.

    ASVspoof rows often start with a speaker id such as ``LA_0001`` and put
    the actual utterance id in another column, for example ``LA_E_1234567``.
    Choosing the first token that starts with ``LA_`` collapses the 2021LA eval
    set to a handful of speakers. Prefer full utterance-id patterns instead.
    """
    cleaned = []
    for token in parts:
        low = token.lower()
        if low in {'-', 'bonafide', 'spoof'}:
            continue
        cleaned.append(Path(token).stem)

    # True ASVspoof utterance ids look like LA_E_xxx, LA_T_xxx, LA_D_xxx, DF_E_xxx.
    for token in cleaned:
        if _ASV_UTT_RE.match(token):
            return token
    for token in cleaned:
        if any(mark in token for mark in ('_E_', '_T_', '_D_')):
            return token

    # Fallbacks for non-standard metadata. Column 1 is commonly the utterance id.
    for idx in [1, 0, 2]:
        if idx < len(parts) and parts[idx].lower() not in {'-', 'bonafide', 'spoof'}:
            return Path(parts[idx]).stem
    return Path(parts[0]).stem


_DF_PHASES = ('progress', 'eval', 'hidden_track')
_DF_PHASE_COL = 7  # ASVspoof2021 keys/CM/trial_metadata.txt: column 7 is the phase.


def _pick_phase(parts: List[str]) -> str:
    """Return the partition tag (progress / eval / hidden_track) of a metadata row."""
    if len(parts) > _DF_PHASE_COL and parts[_DF_PHASE_COL].lower() in _DF_PHASES:
        return parts[_DF_PHASE_COL].lower()
    for token in parts:
        if token.lower() in _DF_PHASES:
            return token.lower()
    return 'unknown'


def load_2021df_eval_list_balanced(meta_path, sample_ratio: float = 1.0, bon_ratio: float = 1.0, seed: int = 1,
                                   phase: str = 'eval'):
    """Load the ASVspoof2021 DF trial list.

    ``phase`` restricts the trials to one official partition of
    ``keys/CM/trial_metadata.txt`` (``progress`` / ``eval`` / ``hidden_track``).
    The default ``eval`` matches the official DF evaluation partition, which is
    also what ``evaluate_2021_*`` uses. Pass ``phase='all'`` (or ``None``) to
    keep every row of the metadata file (previous behaviour).
    """
    meta = Path(meta_path)
    if not meta.exists():
        raise FileNotFoundError(f"2021DF metadata not found: {meta}")

    phase = (phase or 'all').lower()
    if phase not in _DF_PHASES + ('all',):
        raise ValueError(f"phase must be one of {_DF_PHASES + ('all',)}, got {phase!r}")

    keys, labels, attacks = [], {}, {}
    phase_counts: Dict[str, int] = {}
    n_labelled = 0
    with meta.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lab_idx = None
            for i, token in enumerate(parts):
                if token.lower() in {'bonafide', 'spoof'}:
                    lab_idx = i
            if lab_idx is None:
                continue
            n_labelled += 1
            row_phase = _pick_phase(parts)
            phase_counts[row_phase] = phase_counts.get(row_phase, 0) + 1
            if phase != 'all' and row_phase != phase:
                continue
            utt = _pick_utt_id(parts)
            lab = 1 if parts[lab_idx].lower() == 'bonafide' else 0
            attack = 'bonafide' if lab == 1 else (parts[lab_idx - 1] if lab_idx > 0 else 'spoof')
            if utt not in labels:
                keys.append(utt)
            labels[utt] = lab
            attacks[utt] = attack

    dist = ' '.join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
    print(f"[2021DF] metadata={meta} labelled_rows={n_labelled} phase_distribution: {dist}", flush=True)
    if not keys:
        raise RuntimeError(
            f"2021DF phase='{phase}' selected 0 trials from {meta}. "
            f"Available phases in this file: {dist or 'none'}"
        )
    n_bona = sum(1 for k in keys if labels[k] == 1)
    print(
        f"[2021DF] phase='{phase}' selected trials={len(keys)} bonafide={n_bona} spoof={len(keys) - n_bona}",
        flush=True,
    )

    rng = np.random.RandomState(seed)
    bona = [k for k in keys if labels[k] == 1]
    spoof = [k for k in keys if labels[k] == 0]

    def sample(xs, ratio):
        if ratio is None or ratio >= 1.0:
            return list(xs)
        n = max(1, int(round(len(xs) * max(ratio, 0.0))))
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return [xs[i] for i in sorted(idx)]

    # Keep spoof sample_ratio and bonafide bon_ratio, matching common DF subset practice.
    selected = sample(bona, bon_ratio) + sample(spoof, sample_ratio)
    rng.shuffle(selected)
    labels = {k: labels[k] for k in selected}
    attacks = {k: attacks[k] for k in selected}
    n_bona_sel = sum(1 for k in selected if labels[k] == 1)
    print(
        f"[2021DF] after class-ratio sampling (spoof={sample_ratio} bonafide={bon_ratio} seed={seed}): "
        f"N={len(selected)} bonafide={n_bona_sel} spoof={len(selected) - n_bona_sel}",
        flush=True,
    )
    return selected, labels, attacks
