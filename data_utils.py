"""Minimal RawTFNet-compatible protocol utilities.

This keeps the familiar `genSpoof_list` API while avoiding assumptions about
a specific absolute filesystem layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


def genSpoof_list(dir_meta, is_train: bool = False, is_eval: bool = False):
    """Parse ASVspoof protocol files.

    Returns:
        train/dev mode: (labels_dict, utt_id_list), labels: spoof=0, bonafide=1
        eval mode: utt_id_list
    """
    path = Path(dir_meta)
    if not path.exists():
        raise FileNotFoundError(f"Protocol not found: {path}")

    labels: Dict[str, int] = {}
    utts: List[str] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt = parts[1]
            utts.append(utt)
            if is_eval:
                continue
            lab = None
            for token in reversed(parts):
                tok = token.lower()
                if tok in {'bonafide', 'spoof'}:
                    lab = 1 if tok == 'bonafide' else 0
                    break
            if lab is not None:
                labels[utt] = lab

    if is_eval:
        return utts
    return labels, utts
