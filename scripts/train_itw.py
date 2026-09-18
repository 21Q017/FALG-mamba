#!/usr/bin/env python3
"""
Clean training entry for the current best FALG-Mamba-MLF-PG setting.

Default protocol:
  - Train: ASVspoof2019 LA train
  - Validation/checkpoint selection:
      * ASVspoof2021 DF: ONLY the official `eval` partition of keys/CM/trial_metadata.txt
        (--df_eval_phase eval), then class-ratio sampling
      * ASVspoof2021 LA eval subset, attack-balanced sampling + official pooled EER
  - Best checkpoints are saved separately:
      * best_df_model.pt
      * best_la_model.pt

Label convention used by this codebase:
  spoof    -> 0
  bonafide -> 1
Score convention for EER and official score files:
  larger score -> more bonafide
  score = logits[:, 1] - logits[:, 0]
  With --score_mode raw (default) the logits are the RAW un-normalized outputs of the
  linear classifier, so the exported score is unbounded. OC-Softmax is then used only as
  an auxiliary loss on the embedding. --score_mode oc restores the old behaviour where the
  score was the L2-normalized OC cosine similarity, bounded to [-1, 1].
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _sanitize_thread_env(name: str, default: int = 1) -> None:
    value = os.environ.get(name, "")
    try:
        n = int(str(value).strip())
        if n <= 0:
            raise ValueError
    except Exception:
        n = int(default)
    os.environ[name] = str(n)


for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _sanitize_thread_env(_thread_var, default=1)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils import genSpoof_list
from data_utils_2021df import (
    BAD_AUDIO_SENTINEL,
    Dataset_2019LA_train,
    Dataset_2021DF_eval,
    load_2021df_eval_list_balanced,
)
from data_utils_eval_extra import Dataset_Audio_eval
from model_scripts.AA_LG_Mamba_MLF_PG import Model
from itw_validation import build_itw_validation_data


# -----------------------------------------------------------------------------
# Metrics and losses
# -----------------------------------------------------------------------------
_EM_CM = None


def _try_import_cm_metric():
    global _EM_CM
    if _EM_CM is None:
        for mod_name in ("eval_metric_LA", "eval_metrics_DF"):
            try:
                _EM_CM = __import__(mod_name)
                print(f"[eval] using {mod_name}.compute_eer")
                break
            except Exception:
                continue
        if _EM_CM is None:
            _EM_CM = False
            print("[eval] ASVspoof compute_eer not found; using sklearn fallback")
    return _EM_CM


def bonafide_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def compute_eer(labels_arr, scores_arr) -> float:
    labels_arr = np.asarray(labels_arr).astype(int)
    scores_arr = np.asarray(scores_arr, dtype=float)
    target = scores_arr[labels_arr == 1]
    nontarget = scores_arr[labels_arr == 0]
    em = _try_import_cm_metric()
    if em and len(target) > 0 and len(nontarget) > 0:
        return float(em.compute_eer(target, nontarget)[0]) * 100.0

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels_arr, scores_arr, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) * 50.0)


def compute_eer_both_directions(labels_arr, scores_arr):
    return compute_eer(labels_arr, scores_arr), compute_eer(labels_arr, -np.asarray(scores_arr, dtype=float))


def weighted_ce_loss(logits: torch.Tensor, labels: torch.Tensor, w_spoof: float, w_bonafide: float) -> torch.Tensor:
    weight = logits.new_tensor([float(w_spoof), float(w_bonafide)])
    return F.cross_entropy(logits, labels.long(), weight=weight)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.5, max_pairs: int = 4096) -> torch.Tensor:
    scores = bonafide_score_from_logits(logits)
    labels = labels.long().view(-1)
    bon = scores[labels == 1]
    spf = scores[labels == 0]
    if bon.numel() == 0 or spf.numel() == 0:
        return scores.new_zeros(())
    if bon.numel() * spf.numel() > max_pairs:
        nb = max(1, int(max_pairs ** 0.5))
        ns = max(1, int(max_pairs / nb))
        bon = bon[torch.randperm(bon.numel(), device=bon.device)[:min(nb, bon.numel())]]
        spf = spf[torch.randperm(spf.numel(), device=spf.device)[:min(ns, spf.numel())]]
    return F.relu(float(margin) - bon[:, None] + spf[None, :]).mean()


def aux_scale(epoch: int, start_epoch: int, warmup_epochs: int, base_value: float) -> float:
    base_value = float(base_value)
    if base_value == 0.0 or epoch < start_epoch:
        return 0.0
    progress = min(1.0, (epoch - start_epoch + 1) / max(1, warmup_epochs))
    return base_value * progress


def forward_best_loss(model, data, labels, epoch: int, args):
    logits, emb = model(data, return_embedding=True)
    wce = weighted_ce_loss(logits, labels, args.w_spoof, args.w_bonafide)
    lam_rank = aux_scale(epoch, args.aux_start_epoch, args.aux_warmup_epochs, args.rank_lambda)
    lam_oc = aux_scale(epoch, args.aux_start_epoch, args.aux_warmup_epochs, args.oc_lambda)
    if not hasattr(model, "oc_head"):
        raise RuntimeError("Current best loss requires model.oc_head. Keep --use_oc enabled.")
    loss = wce + lam_oc * model.oc_head(emb, labels)
    if lam_rank != 0.0:
        rank = pairwise_rank_loss(logits, labels, margin=args.rank_margin, max_pairs=args.rank_max_pairs)
        loss = loss + lam_rank * rank
    return loss, logits


# -----------------------------------------------------------------------------
# Data parsing and sampling
# -----------------------------------------------------------------------------
_ASV_UTT_RE = re.compile(r"^(?:LA|DF)_[A-Z]_[0-9]+$", re.IGNORECASE)


def _audio_stems(audio_dir: Path, allowed_exts=(".flac", ".wav", ".mp3")):
    stems = set()
    if not audio_dir.is_dir():
        return stems
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            stems.add(p.stem)
    return stems


def _candidate_utt_ids(parts: List[str]):
    raw = []
    for token in parts:
        tok = str(token).strip().strip(",")
        if not tok:
            continue
        if tok.lower() in {"-", "bonafide", "spoof", "notrim", "trim"}:
            continue
        raw.append(Path(tok).stem)
    high, mid, low = [], [], []
    for tok in raw:
        if _ASV_UTT_RE.match(tok):
            high.append(tok)
        elif any(mark in tok for mark in ("_E_", "_T_", "_D_")):
            mid.append(tok)
        else:
            low.append(tok)
    return high + mid + low


def _pick_utt_id_with_audio(parts: List[str], audio_stems=None):
    cands = _candidate_utt_ids(parts)
    if audio_stems:
        for tok in cands:
            if tok in audio_stems:
                return tok
    return cands[0] if cands else Path(parts[1] if len(parts) > 1 else parts[0]).stem


_PHASES = ("progress", "eval", "hidden_track")
_PHASE_COL = 7  # ASVspoof2021 keys/CM/trial_metadata.txt: column 7 holds the phase.


def _row_phase(parts: List[str]) -> str:
    if len(parts) > _PHASE_COL and parts[_PHASE_COL].lower() in _PHASES:
        return parts[_PHASE_COL].lower()
    for token in parts:
        if token.lower() in _PHASES:
            return token.lower()
    return "unknown"


def load_2021la_metadata(meta_path, audio_dir, debug_name="2021LA", phase="eval"):
    """Load 2021LA trials, restricted to one official partition.

    The old version read every row (progress + eval + hidden_track) and sampled
    from the union, while the official pooled EER only merges rows whose phase
    equals `la_eval_phase`. The subset that actually reached the official metric was
    therefore much smaller than the reported N, and the quick EER was computed over
    a mixture of partitions. Filtering here keeps both metrics on the same trials.
    """
    meta = Path(meta_path)
    audio_dir = Path(audio_dir)
    if not meta.exists():
        raise FileNotFoundError(f"2021LA metadata not found: {meta}")
    phase = (phase or "all").lower()
    if phase not in _PHASES + ("all",):
        raise ValueError(f"phase must be one of {_PHASES + ('all',)}, got {phase!r}")
    stems = _audio_stems(audio_dir)
    print(f"[{debug_name}] audio_dir={audio_dir} audio_files={len(stems)}")
    keys, labels, attacks = [], {}, {}
    line_count = labelled_count = duplicate_count = 0
    phase_counts = defaultdict(int)
    head_examples = []
    with meta.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_count += 1
            parts = line.strip().split()
            if not parts:
                continue
            label_idx = None
            for i, token in enumerate(parts):
                if token.lower() in {"bonafide", "spoof"}:
                    label_idx = i
            if label_idx is None:
                continue
            labelled_count += 1
            row_phase = _row_phase(parts)
            phase_counts[row_phase] += 1
            if phase != "all" and row_phase != phase:
                continue
            utt = _pick_utt_id_with_audio(parts, stems)
            lab = 1 if parts[label_idx].lower() == "bonafide" else 0
            attack = "bonafide" if lab == 1 else (parts[label_idx - 1] if label_idx > 0 else "spoof")
            if utt in labels:
                duplicate_count += 1
            keys.append(utt)
            labels[utt] = lab
            attacks[utt] = attack
            if len(head_examples) < 5:
                head_examples.append((line_count, utt, "bonafide" if lab == 1 else "spoof", parts[:10]))

    seen, uniq = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            uniq.append(k)

    dist = " ".join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
    print(f"[{debug_name}] meta_lines={line_count} labelled={labelled_count} phase_distribution: {dist}")
    print(f"[{debug_name}] phase='{phase}' unique_keys={len(uniq)} duplicates_removed={duplicate_count}")
    if not uniq:
        raise RuntimeError(f"2021LA phase='{phase}' selected 0 trials from {meta}. Available: {dist or 'none'}")
    if head_examples:
        print(f"[{debug_name}] first parsed rows:")
        for lineno, utt, lab, parts in head_examples:
            print(f"  line={lineno} utt={utt} label={lab} parts={parts}")
    if stems:
        missing = [k for k in uniq[:1000] if k not in stems]
        if missing:
            print(f"[{debug_name}] warning: first-1000 missing_audio={len(missing)} examples={missing[:8]}")
        else:
            print(f"[{debug_name}] first-1000 keys all have matching audio files")
    return uniq, labels, attacks


def sample_eval_keys_class_ratio(keys, labels, sample_ratio=1.0, bon_ratio=1.0, seed=1):
    rng = np.random.RandomState(seed)
    bona = [k for k in keys if labels[k] == 1]
    spoof = [k for k in keys if labels[k] == 0]

    def sample(xs, ratio):
        if ratio is None or ratio >= 1.0:
            return list(xs)
        n = max(1, int(round(len(xs) * max(ratio, 0.0))))
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return [xs[i] for i in sorted(idx)]

    selected = sample(bona, bon_ratio) + sample(spoof, sample_ratio)
    rng.shuffle(selected)
    return selected, {k: labels[k] for k in selected}, {k: "bonafide" if labels[k] == 1 else "spoof" for k in selected}


def sample_eval_keys_attack_balanced(keys, labels, attacks, sample_ratio=0.3, seed=1234):
    rng = np.random.RandomState(seed)
    bona = [k for k in keys if labels[k] == 1]
    spoof_groups = defaultdict(list)
    for k in keys:
        if labels[k] == 0:
            spoof_groups[attacks.get(k, "spoof")].append(k)

    total_target = max(1, int(len(keys) * float(sample_ratio)))
    bon_target = min(len(bona), max(1, int(total_target * 0.1)))
    spoof_target = max(1, total_target - bon_target)

    selected_bona = list(rng.choice(bona, size=bon_target, replace=False)) if len(bona) > bon_target else list(bona)
    selected_spoof = []
    groups = sorted(spoof_groups.items())
    if groups:
        base = spoof_target // len(groups)
        rem = spoof_target % len(groups)
        for i, (_attack, items) in enumerate(groups):
            n = min(len(items), base + (1 if i < rem else 0))
            if n > 0:
                selected_spoof.extend(list(rng.choice(items, size=n, replace=False)))
    selected = selected_bona + selected_spoof
    rng.shuffle(selected)
    return selected, {k: labels[k] for k in selected}, {k: attacks[k] for k in selected}


def print_selected_samples(name, keys, labels, attacks, audio_dir, ext=".flac", n=5, out_fold=None):
    if n <= 0:
        return
    audio_dir = Path(audio_dir)
    by_label = [("bonafide", 1), ("spoof", 0)]
    rows = []
    for label_name, label_value in by_label:
        shown = 0
        print(f"[{name}] selected {label_name} examples shown={n}")
        for k in keys:
            if labels[k] != label_value:
                continue
            path = audio_dir / f"{k}{ext}"
            print(f"  [{label_name} #{shown+1}] utt={k} attack={attacks.get(k, '-')} exists={path.exists()} path={path}")
            rows.append([k, label_name, attacks.get(k, "-"), str(path), str(path.exists())])
            shown += 1
            if shown >= n:
                break
    if out_fold:
        out_dir = Path(out_fold) / "selected_eval_subsets"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{name}_selected_samples.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["utt", "label", "attack", "path", "exists"])
            writer.writerows(rows)
        print(f"[{name}] selected sample list saved: {out_csv}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def autocast_context(args, device):
    if not (args.amp and str(device).startswith("cuda")):
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _infer_2021la_keys_dir(args):
    if args.data_2021la_keys_dir:
        return Path(args.data_2021la_keys_dir)
    meta = Path(args.data_2021la_meta)
    if meta.name == "trial_metadata.txt" and meta.parent.name.upper() == "CM":
        return meta.parent.parent
    return Path("/root/autodl-tmp/data/2021LA/keys")


def compute_2021la_official_from_scores(keys, scores, args, desc="2021LA", epoch=None):
    keys_dir = _infer_2021la_keys_dir(args)
    cm_key_file = keys_dir / "CM" / "trial_metadata.txt"
    if not cm_key_file.exists():
        print(f"  [2021LA official] skip: CM key not found: {cm_key_file}")
        return None, None, None

    score_dir = Path(args.out_fold) / "eval_scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    safe_desc = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(desc)).strip("_")
    # Keeping one score file per epoch writes ~60 files of tens of thousands of lines.
    # Only do that when explicitly requested; otherwise reuse a single scratch file.
    ep = f"epoch{int(epoch):03d}_" if (epoch is not None and args.save_epoch_scores) else ""
    score_file = score_dir / f"scores_{ep}{safe_desc}_{args.la_eval_phase}.txt"
    with score_file.open("w", encoding="utf-8") as f:
        for utt_id, score in zip(keys, scores):
            f.write(f"{Path(str(utt_id)).stem} {float(score):.8f}\n")

    official_eer = None
    min_tDCF = None
    merged_n = 0
    try:
        import eval_metric_LA as em_la
        cm_data = pd.read_csv(cm_key_file, sep=" ", header=None)
        submission_scores = pd.read_csv(score_file, sep=" ", header=None, skipinitialspace=True)
        cm_scores = submission_scores.merge(cm_data[cm_data[7] == args.la_eval_phase], left_on=0, right_on=1, how="inner")
        merged_n = int(len(cm_scores))
        if merged_n == 0:
            raise RuntimeError("score file has no overlap with 2021LA CM metadata")
        bona_cm = cm_scores[cm_scores[5] == "bonafide"]["1_x"].values
        spoof_cm = cm_scores[cm_scores[5] == "spoof"]["1_x"].values
        official_eer = float(em_la.compute_eer(bona_cm, spoof_cm)[0]) * 100.0
    except Exception as exc:
        print(f"  [2021LA official] pooled EER failed: {exc}")

    try:
        import evaluate_2021_LA as e2021la
        e2021la.phase = args.la_eval_phase
        e2021la.asv_key_file = str(keys_dir / "ASV" / "trial_metadata.txt")
        e2021la.asv_scr_file = str(keys_dir / "ASV" / "ASVTorch_Kaldi" / "score.txt")
        e2021la.cm_key_file = str(cm_key_file)
        if Path(e2021la.asv_key_file).exists() and Path(e2021la.asv_scr_file).exists():
            min_tDCF = float(e2021la.eval_to_score_file(str(score_file), str(cm_key_file)))
    except Exception as exc:
        print(f"  [2021LA official] min-tDCF failed: {exc}")

    if official_eer is not None:
        mt = f", min t-DCF={min_tDCF:.4f}" if min_tDCF is not None else ""
        print(f"  [2021LA official] pooled EER={official_eer:.4f}%{mt} merged={merged_n}/{len(keys)} score_file={score_file}")
    return official_eer, min_tDCF, str(score_file)


def report_unreadable(utts, desc: str, args):
    """Log the trials that had to be dropped because their audio would not decode."""
    uniq = sorted(set(utts))
    out = Path(args.out_fold) / "unreadable_files.txt"
    existing = set()
    if out.exists():
        existing = {ln.split()[0] for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()}
    new = [u for u in uniq if u not in existing]
    if new:
        with out.open("a", encoding="utf-8") as f:
            for u in new:
                f.write(f"{u}\t{desc}\n")
    print(
        f"  [{desc}][WARN] dropped {len(uniq)} undecodable trial(s) from the metric "
        f"(e.g. {uniq[:3]}). Full list: {out}. "
        f"Re-download them, or pass --exclude_utts to remove them from the trial list.",
        flush=True,
    )


def load_exclude_utts(path) -> set:
    """Read utterance ids to drop from the trial lists (first whitespace field per line)."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--exclude_utts file not found: {p}")
    utts = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        utts.add(Path(line.split()[0]).stem)
    print(f"[exclude] loaded {len(utts)} utterance ids to exclude from {p}")
    return utts


def evaluate(model, loader, device, labels_dict: Dict[str, int], attacks_dict: Dict[str, str], desc: str, args, epoch=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    total, correct = 0, 0
    keys_all, labels_all, scores_all = [], [], []

    unreadable = []
    with torch.no_grad():
        for data, keys in tqdm(loader, desc=desc, ncols=110, leave=False):
            keys = [str(k) for k in keys]
            # Trials whose audio could not be decoded come back tagged with a sentinel.
            # Scoring the zero-filled placeholder would silently corrupt the EER, so the
            # whole trial is dropped (and reported) instead.
            keep = [i for i, k in enumerate(keys) if not k.startswith(BAD_AUDIO_SENTINEL)]
            unreadable.extend(k[len(BAD_AUDIO_SENTINEL):] for k in keys if k.startswith(BAD_AUDIO_SENTINEL))
            if not keep:
                continue
            data = data.to(device, non_blocking=True)
            with autocast_context(args, device):
                out = model(data)
                logits = out[0] if isinstance(out, tuple) else out
            if len(keep) != len(keys):
                idx = torch.tensor(keep, device=logits.device)
                logits = logits.index_select(0, idx)
                keys = [keys[i] for i in keep]
            target = torch.tensor([labels_dict[k] for k in keys], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits.float(), target)
            score = bonafide_score_from_logits(logits).detach().float().cpu().numpy()
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(target).sum().item())
            total += int(target.numel())
            total_loss += float(loss.item())
            n_batches += 1
            keys_all.extend(keys)
            labels_all.extend(target.detach().cpu().numpy().tolist())
            scores_all.extend(score.tolist())

    if unreadable:
        report_unreadable(unreadable, desc, args)

    labels_arr = np.asarray(labels_all, dtype=int)
    scores_arr = np.asarray(scores_all, dtype=float)
    quick_eer, eer_inv = compute_eer_both_directions(labels_arr, scores_arr)
    selected_eer = quick_eer
    official_eer = min_tDCF = None
    if "2021LA" in desc and args.la_official_eval:
        official_eer, min_tDCF, _ = compute_2021la_official_from_scores(keys_all, scores_all, args, desc=desc, epoch=epoch)
        if official_eer is not None:
            selected_eer = official_eer

    bon_scores = scores_arr[labels_arr == 1]
    spoof_scores = scores_arr[labels_arr == 0]
    separation = float(bon_scores.mean() - spoof_scores.mean()) if len(bon_scores) and len(spoof_scores) else float("nan")
    stats = {"eer_inv": eer_inv, "separation": separation, "official_eer": official_eer, "quick_eer": quick_eer, "min_tDCF": min_tDCF}

    per_attack = {}
    for attack in sorted(set(attacks_dict.values())):
        idx = [i for i, k in enumerate(keys_all) if attacks_dict.get(k) == attack]
        if attack == "bonafide" or not idx:
            continue
        sub_labels = np.concatenate([np.ones(len(bon_scores), dtype=int), np.zeros(len(idx), dtype=int)])
        sub_scores = np.concatenate([bon_scores, scores_arr[idx]])
        if len(np.unique(sub_labels)) == 2:
            per_attack[attack] = compute_eer(sub_labels, sub_scores)

    return total_loss / max(n_batches, 1), 100.0 * correct / max(total, 1), selected_eer, per_attack, stats


def produce_scores(model, loader, device, out_path, args):
    model.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for data, keys in tqdm(loader, desc=f"score->{Path(out_path).name}", ncols=110, leave=False):
            data = data.to(device, non_blocking=True)
            with autocast_context(args, device):
                out = model(data)
                logits = out[0] if isinstance(out, tuple) else out
            score = bonafide_score_from_logits(logits).detach().float().cpu().numpy()
            for k, s in zip(keys, score):
                k = str(k)
                if k.startswith(BAD_AUDIO_SENTINEL):
                    continue  # unreadable trial: emit no score rather than a fake one
                f.write(f"{k} {float(s):.8f}\n")


def evaluate_itw(model, loader, device, labels_dict: Dict[str, int], desc: str, args, return_scores: bool = False):
    """Evaluate deterministic multi-crop ITW trials and average crop logits per utterance."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    total, correct = 0, 0
    keys_all, labels_all, scores_all = [], [], []
    unreadable = []

    with torch.inference_mode():
        for data, keys, valid in tqdm(loader, desc=desc, ncols=110, leave=False):
            valid_mask = torch.as_tensor(valid, dtype=torch.bool)
            keys = [str(k) for k in keys]
            if not bool(valid_mask.any()):
                unreadable.extend(keys)
                continue
            if not bool(valid_mask.all()):
                unreadable.extend(keys[i] for i in range(len(keys)) if not bool(valid_mask[i]))
                data = data[valid_mask]
                keys = [keys[i] for i in range(len(keys)) if bool(valid_mask[i])]

            # Dataset returns [B, C, T], where C is the deterministic crop count.
            if data.ndim != 3:
                raise RuntimeError(f"ITW batch must have shape [B,crops,time], got {tuple(data.shape)}")
            batch, crops, length = data.shape
            flat = data.reshape(batch * crops, length).to(device, non_blocking=True)
            with autocast_context(args, device):
                out = model(flat)
                logits = out[0] if isinstance(out, tuple) else out
            logits = logits.reshape(batch, crops, -1).mean(dim=1)
            target = torch.tensor([labels_dict[k] for k in keys], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits.float(), target)
            score = bonafide_score_from_logits(logits).detach().float().cpu().numpy()
            pred = logits.argmax(dim=1)

            correct += int(pred.eq(target).sum().item())
            total += int(target.numel())
            total_loss += float(loss.item())
            n_batches += 1
            keys_all.extend(keys)
            labels_all.extend(target.detach().cpu().numpy().tolist())
            scores_all.extend(score.tolist())

    if unreadable:
        report_unreadable(unreadable, desc, args)
    if not labels_all:
        raise RuntimeError("ITW validation scored zero readable trials")

    labels_arr = np.asarray(labels_all, dtype=int)
    scores_arr = np.asarray(scores_all, dtype=float)
    eer, eer_inv = compute_eer_both_directions(labels_arr, scores_arr)
    bona = scores_arr[labels_arr == 1]
    spoof = scores_arr[labels_arr == 0]
    separation = float(bona.mean() - spoof.mean())
    stats = {
        "eer_inv": eer_inv,
        "separation": separation,
        "n_scored": int(labels_arr.size),
        "n_bonafide": int((labels_arr == 1).sum()),
        "n_spoof": int((labels_arr == 0).sum()),
        "n_unreadable": len(set(unreadable)),
    }
    if return_scores:
        stats["keys"] = keys_all
        stats["labels"] = labels_arr
        stats["scores"] = scores_arr
    return total_loss / max(n_batches, 1), 100.0 * correct / max(total, 1), eer, {}, stats


def produce_itw_scores(stats, out_path):
    if not all(name in stats for name in ("keys", "labels", "scores")):
        raise ValueError("ITW score export requires evaluate_itw(..., return_scores=True)")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path = out_path.with_suffix(".tsv")
    with out_path.open("w", encoding="utf-8") as f:
        for key, score in zip(stats["keys"], stats["scores"]):
            f.write(f"{key} {float(score):.10f}\n")
    with detail_path.open("w", encoding="utf-8") as f:
        f.write("utt\tlabel\tscore\n")
        for key, label, score in zip(stats["keys"], stats["labels"], stats["scores"]):
            name = "bonafide" if int(label) == 1 else "spoof"
            f.write(f"{key}\t{name}\t{float(score):.10f}\n")
    print(f"[ITW] scores saved: {out_path}")
    print(f"[ITW] labeled scores saved: {detail_path}")


# -----------------------------------------------------------------------------
# Model, loaders, logging
# -----------------------------------------------------------------------------
def make_loader(dataset, args, train: bool = False, shuffle: bool = False, drop_last: bool = False):
    batch_size = int(args.batch_size if train else args.eval_batch_size)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def build_model(args, device):
    d_args = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "n_layer": args.n_layer,
        "num_classes": 2,
        "n_query": args.n_query,
        "scan": args.scan,
        "use_oc": True,
        # 'cls' -> raw un-normalized logits; 'oc' -> normalized OC cosine score.
        "logit_source": "cls" if getattr(args, "score_mode", "raw") == "raw" else "oc",
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
    model = Model(d_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: AA_LG_Mamba_MLF_PG(best-clean) params={n_params/1e6:.3f}M device={device}")
    m = model.module if hasattr(model, "module") else model
    gamma = float(m.get_prior_strength())
    print(
        f"[MLF] fusion={args.mlf_fusion} align_level={args.mlf_align_level} "
        f"prior_strength(gamma)={gamma:.3f} preact_fix={args.preact_fix} freq_flatten={args.freq_flatten}"
    )
    if d_args["logit_source"] == "cls":
        print("[score] mode=raw  logits=nn.Linear(cls) (un-normalized); OC-Softmax kept as auxiliary embedding loss only")
    else:
        print("[score] mode=oc   logits=OC cosine similarity (L2-normalized, bounded to [-1,1])")
    try:
        import mamba_ssm
        print(f"[mamba_ssm] using: {getattr(mamba_ssm, '__file__', 'unknown')}")
    except Exception:
        pass
    return model


def load_initial_checkpoint(model, ckpt_path: str, device):
    """Load model weights for fine-tuning while intentionally resetting optimizer state."""
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"--init_ckpt not found: {path}")
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        state = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload).__name__}")
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    print(f"[finetune] loaded initial weights: {path}")
    print("[finetune] optimizer/scheduler are reset on purpose; this is fine-tuning, not exact resume")


def build_optimizer(model, args):
    """Discriminative learning rates: frozen/slow front, normal trunk, flexible head."""
    groups = {"front": [], "trunk": [], "head": []}
    head_roots = ("pool.", "embed.", "cls.", "oc_head.")
    for name, param in model.named_parameters():
        if name.startswith("front."):
            groups["front"].append(param)
        elif name.startswith(head_roots):
            groups["head"].append(param)
        else:
            groups["trunk"].append(param)
    multipliers = {
        "front": float(args.front_lr_mult),
        "trunk": float(args.trunk_lr_mult),
        "head": float(args.head_lr_mult),
    }
    param_groups = []
    for name in ("front", "trunk", "head"):
        params = groups[name]
        if not params:
            continue
        lr = float(args.base_lr) * multipliers[name]
        param_groups.append({"params": params, "lr": lr, "initial_lr": lr, "group_name": name})
        n = sum(p.numel() for p in params)
        print(f"[optim] group={name:<5s} params={n/1e3:.1f}K lr={lr:.3e} (x{multipliers[name]:g})")
    return optim.Adam(param_groups, betas=(0.9, 0.98), eps=1e-9, weight_decay=float(args.weight_decay))


def configure_finetune_stage(model, epoch: int, args):
    """Optionally freeze the expensive convolutional front end for the first few FT epochs."""
    m = model.module if hasattr(model, "module") else model
    freeze = bool(args.init_ckpt) and int(epoch) <= int(args.freeze_front_epochs)
    for p in m.front.parameters():
        p.requires_grad_(not freeze)
    if freeze:
        # Keep BatchNorm running statistics fixed as well as the parameters.
        m.front.eval()
    if epoch == 1 or epoch == int(args.freeze_front_epochs) + 1:
        state = "frozen" if freeze else "trainable"
        print(f"[finetune] epoch={epoch}: front end is {state}")
    return freeze


def joint_selection_value(df_eer: float, la_eer: float, df_weight: float) -> float:
    w = min(1.0, max(0.0, float(df_weight)))
    return w * float(df_eer) + (1.0 - w) * float(la_eer)


def collect_mlf_weight_stats(model, loader, device, max_batches: int):
    m = model.module if hasattr(model, "module") else model
    if max_batches <= 0 or not hasattr(m, "get_mlf_weight_info"):
        return None
    was_training = model.training
    model.eval()
    sums = None
    n_batches = 0
    with torch.no_grad():
        for data, _keys in loader:
            data = data.to(device)
            _ = model(data)
            alpha = getattr(m, "last_level_alpha", None)
            if alpha is None:
                continue
            if alpha.dim() == 5:
                a = alpha.mean(dim=(0, 2, 3, 4)).detach().cpu()
            else:
                continue
            sums = a if sums is None else sums + a
            n_batches += 1
            if n_batches >= max_batches:
                break
    if was_training:
        model.train()
    info = m.get_mlf_weight_info()
    dyn = (sums / max(n_batches, 1)) if sums is not None else None
    return {"names": info["names"], "prior": info["prior"].float().cpu(), "prior_strength": float(info["prior_strength"]), "dynamic_alpha_mean": dyn, "n_batches": n_batches}


def format_mlf_stats(stats):
    if stats is None:
        return None
    names = stats["names"]
    msg = "prior " + "  ".join(f"{n}={float(v):.3f}" for n, v in zip(names, stats["prior"]))
    msg += f"  gamma={stats['prior_strength']:.3f}"
    dyn = stats.get("dynamic_alpha_mean")
    if dyn is not None:
        msg += "  |  dyn-alpha " + "  ".join(f"{n}={float(v):.3f}" for n, v in zip(names, dyn))
        msg += f"  (batches={stats['n_batches']})"
    return msg


def write_mlf_stats_csv(path, epoch, stats):
    if stats is None:
        return
    names = list(stats["names"])
    dyn = stats.get("dynamic_alpha_mean")
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            header = ["epoch"] + [f"prior_{n}" for n in names] + ["prior_strength"]
            if dyn is not None:
                header += [f"dyn_alpha_{n}" for n in names]
            header += ["n_batches"]
            f.write(",".join(header) + "\n")
        row = [str(epoch)] + [f"{float(v):.6f}" for v in stats["prior"]] + [f"{stats['prior_strength']:.6f}"]
        if dyn is not None:
            row += [f"{float(v):.6f}" for v in dyn]
        row += [str(stats["n_batches"])]
        f.write(",".join(row) + "\n")


def parse_int_list_csv(text: str):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)          # RawBoost algo choice uses random.choice
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _drop_excluded(keys, labels, attacks, excluded, name):
    if not excluded:
        return keys, labels, attacks
    kept = [k for k in keys if k not in excluded]
    n_drop = len(keys) - len(kept)
    if n_drop:
        print(f"[{name}] excluded {n_drop} trial(s) listed in --exclude_utts; N {len(keys)} -> {len(kept)}")
    return kept, {k: labels[k] for k in kept}, {k: attacks[k] for k in kept}


def build_2021df_eval(args, excluded=None):
    keys, labels, attacks = load_2021df_eval_list_balanced(
        args.trial_metadata,
        sample_ratio=args.sample_ratio,
        bon_ratio=args.bon_ratio,
        seed=args.df_eval_sampling_seed,
        phase=args.df_eval_phase,
    )
    keys, labels, attacks = _drop_excluded(keys, labels, attacks, excluded, "2021DF-eval-subset")
    print_selected_samples("2021DF-eval-subset", keys, labels, attacks, args.data_2021df_flac, n=args.show_sample_examples, out_fold=args.out_fold)
    dataset = Dataset_2021DF_eval(keys, args.data_2021df_flac, name="2021DF-eval-subset", on_error=args.audio_error)
    return make_loader(dataset, args, train=False), labels, attacks


def build_2021la_eval(args, excluded=None):
    keys, labels, attacks = load_2021la_metadata(
        args.data_2021la_meta, args.data_2021la_flac,
        debug_name="2021LA-eval-subset", phase=args.la_eval_phase,
    )
    keys, labels, attacks = sample_eval_keys_attack_balanced(
        keys, labels, attacks,
        sample_ratio=args.la_eval_sampling_ratio,
        seed=args.la_eval_sampling_seed,
    )
    keys, labels, attacks = _drop_excluded(keys, labels, attacks, excluded, "2021LA-eval-subset")
    n_bona = sum(1 for k in keys if labels[k] == 1)
    n_spoof = sum(1 for k in keys if labels[k] == 0)
    print(f"[2021LA-eval-subset] sampled subset N={len(keys)} bonafide={n_bona} spoof={n_spoof} strategy=attack_balanced total_ratio={args.la_eval_sampling_ratio} seed={args.la_eval_sampling_seed}")
    print_selected_samples("2021LA-eval-subset", keys, labels, attacks, args.data_2021la_flac, n=args.show_sample_examples, out_fold=args.out_fold)
    dataset = Dataset_Audio_eval(keys, args.data_2021la_flac, ext=".flac", target_len=args.eval_len, name="2021LA-eval-subset", on_error=args.audio_error)
    return make_loader(dataset, args, train=False), labels, attacks


def build_itw_eval(args):
    dataset, labels, protocol = build_itw_validation_data(args)
    kwargs = dict(
        batch_size=int(args.itw_eval_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(dataset, **kwargs)
    return loader, labels, protocol


def train_epoch(model, loader, optimizer, scheduler, device, epoch, best_eer, args, scaler=None):
    model.train()
    configure_finetune_stage(model, epoch, args)
    total_loss = 0.0
    correct = total = 0
    train_scores, train_labels = [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=110)
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(args, device):
            loss, logits = forward_best_loss(model, data, target, epoch, args)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int(pred.eq(target).sum().item())
        total += int(target.size(0))
        with torch.no_grad():
            score = bonafide_score_from_logits(logits).detach().cpu().numpy()
            train_scores.extend(score.tolist())
            train_labels.extend(target.detach().cpu().tolist())
        if batch_idx % 50 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss/(batch_idx+1):.4f}",
                "acc": f"{100.0*correct/max(total,1):.1f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "best_val": f"{best_eer:.2f}%" if best_eer < float("inf") else "-",
            })
    train_eer = compute_eer(np.array(train_labels), np.array(train_scores))
    return total_loss / max(len(loader), 1), train_eer


def parse_args():
    parser = argparse.ArgumentParser(description="Clean best FALG-Mamba-MLF-PG trainer")
    parser.add_argument("-o", "--out_fold", type=str, default="./exp_mlf_pg_attn_rb0123_best_clean")
    parser.add_argument("--data_2019la", type=str, default="/root/autodl-tmp/data/2019LA")
    parser.add_argument("--data_2021df_flac", type=str, default="/root/autodl-tmp/data/2021DF/ASVspoof2021_DF_eval/flac")
    parser.add_argument("--trial_metadata", type=str, default="/root/autodl-tmp/data/2021DF/keys/CM/trial_metadata.txt")
    parser.add_argument("--data_2021la_flac", type=str, default="/root/autodl-tmp/data/2021LA/ASVspoof2021_LA_eval/flac")
    parser.add_argument("--data_2021la_meta", type=str, default="/root/autodl-tmp/data/2021LA/keys/CM/trial_metadata.txt")
    parser.add_argument("--data_2021la_keys_dir", type=str, default="")

    # ITW-only validation mode: train on 2019LA and evaluate/save only on ITW.
    parser.add_argument("--itw_only_val", action="store_true", default=False,
                        help="skip 2021DF/LA and use only ITW for per-epoch validation/checkpoint selection")
    # Optional In-the-Wild validation. Disabled by default so existing runs are unchanged.
    parser.add_argument("--use_itw_val", action="store_true", default=False,
                        help="evaluate In-the-Wild during training and log ITW EER")
    parser.add_argument("--itw_root", type=str, default="~/autodl-tmp/data/release_in_the_wild")
    parser.add_argument("--itw_protocol", type=str, default="",
                        help="ITW protocol/metadata path; auto-discovered under --itw_root when omitted")
    parser.add_argument("--itw_real_label", type=int, default=1, choices=[0, 1],
                        help="for numeric protocol labels, which value denotes bonafide")
    parser.add_argument("--itw_sample_ratio", type=float, default=1.0,
                        help="fixed class-stratified ITW validation fraction in (0,1]")
    parser.add_argument("--itw_sampling_seed", type=int, default=1234,
                        help="ITW subset seed, independent of the training seed")
    parser.add_argument("--itw_num_crops", type=int, default=1,
                        help="deterministic evenly-spaced crops per ITW utterance; crop logits are averaged")
    parser.add_argument("--itw_eval_batch_size", type=int, default=64,
                        help="ITW utterances per batch; effective model batch is this value times --itw_num_crops")
    parser.add_argument("--itw_eval_interval", type=int, default=1,
                        help="run ITW validation every N epochs (and always on the final epoch)")
    parser.add_argument("--itw_monitor_only", action="store_true", default=False,
                        help="log ITW but do not save best_itw_model.pt or use ITW for joint selection")
    parser.add_argument("--joint_itw_weight", type=float, default=0.0,
                        help="ITW weight for a separate best_joint_itw checkpoint; 0 keeps DF/LA-only joint selection")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_query", type=int, default=4)
    parser.add_argument("--scan", type=str, default="alt", choices=["alt", "uni", "bi"])
    parser.add_argument("--sinc_channels", type=int, default=70)
    parser.add_argument("--pool_heads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freq_pool", type=int, default=3)
    # Cross-level fusion: which level's time grid the other levels are aligned onto.
    # The three levels have very different lengths (~597 / 99 / 16 frames at 64600 samples),
    # so 'deep' collapses the shallow level ~37x. 'middle' is the new default.
    parser.add_argument("--mlf_align_level", type=str, default="middle",
                        choices=["shallow", "middle", "deep"])
    parser.add_argument("--mlf_fusion", type=str, default="attn",
                        choices=["attn", "prior_attn", "prior_only", "mean"])
    parser.add_argument("--mlf_prior_init", type=str, default="0.33,0.33,0.34")
    parser.add_argument("--mlf_prior_strength", type=float, default=0.0)
    parser.add_argument("--freq_flatten", dest="freq_flatten", action="store_true", default=False)
    parser.add_argument("--flatten_mode", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--freq_tokenizer", type=str, default="k_query",
                        choices=["k_query", "fixed_band"],
                        help="Exp B: k_query (default) | fixed_band (equal-width sub-bands, param-free)")
    # Pre-activation fix in the residual blocks (legacy RawNet2 baseline discards it).
    parser.add_argument("--preact_fix", dest="preact_fix", action="store_true", default=True)
    parser.add_argument("--no_preact_fix", dest="preact_fix", action="store_false")
    parser.add_argument("--oc_alpha", type=float, default=20.0)
    parser.add_argument("--oc_m_real", type=float, default=0.9)
    parser.add_argument("--oc_m_fake", type=float, default=0.2)

    parser.add_argument("--w_spoof", type=float, default=0.1)
    parser.add_argument("--w_bonafide", type=float, default=0.9)
    parser.add_argument("--rank_lambda", type=float, default=0.0)
    parser.add_argument("--rank_margin", type=float, default=0.0)
    parser.add_argument("--rank_max_pairs", type=int, default=4096)
    parser.add_argument("--oc_lambda", type=float, default=0.02)
    parser.add_argument("--aux_start_epoch", type=int, default=0)
    parser.add_argument("--aux_warmup_epochs", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0,
                        help="final cosine LR as a fraction of each parameter group's initial LR")
    parser.add_argument("--warmup_start_ratio", type=float, default=0.0,
                        help="LR fraction at the first warmup step; 0.1 is usually safer for fine-tuning")

    # Fine-tuning from an existing best checkpoint. The architecture arguments must match.
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="model checkpoint used to initialize a new low-LR fine-tuning run")
    parser.add_argument("--freeze_front_epochs", type=int, default=0,
                        help="freeze front-end parameters and BN statistics for the first N fine-tuning epochs")
    parser.add_argument("--front_lr_mult", type=float, default=1.0)
    parser.add_argument("--trunk_lr_mult", type=float, default=1.0)
    parser.add_argument("--head_lr_mult", type=float, default=1.0)
    parser.add_argument("--save_epoch_ckpts", action="store_true", default=False,
                        help="save checkpoint/epoch_XXX.pt for later checkpoint averaging")
    parser.add_argument("--resume_dir", type=str, default="",
                        help="resume an interrupted run: dir whose checkpoint/epoch_<N>.pt continues; "
                             "requires --resume_from_epoch>=1. Loads weights, warm-starts cosine LR to "
                             "that epoch's step, keeps best-ITW history from train_log.csv, appends rows. "
                             "Does NOT reset optimizer adaptive state (only Adam's m/v restart).")
    parser.add_argument("--resume_from_epoch", type=int, default=0,
                        help="epoch already completed when --resume_dir set; training continues at this+1")
    parser.add_argument("--joint_df_weight", type=float, default=0.7,
                        help="weight of DF EER in the single-checkpoint joint selection score")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--save_epoch_scores", action="store_true", default=False)
    # Corrupt/undecodable audio: 'skip' drops the trial (and logs it) instead of killing the run.
    parser.add_argument("--audio_error", type=str, default="skip", choices=["skip", "raise"])
    parser.add_argument("--exclude_utts", type=str, default="",
                        help="file listing utterance ids to drop (e.g. output of scripts/scan_audio.py)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=True)
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")

    # Score convention: 'raw' exports the un-normalized classifier logit difference.
    # 'oc' restores the legacy behaviour where the score is the normalized OC cosine similarity.
    parser.add_argument("--score_mode", type=str, default="raw", choices=["raw", "oc"])

    parser.add_argument("--rawboost_algos", type=str, default="1,2,3")
    # Only the official 2021DF 'eval' partition is used for validation by default.
    parser.add_argument("--df_eval_phase", type=str, default="eval",
                        choices=["progress", "eval", "hidden_track", "all"])
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--bon_ratio", type=float, default=0.2)
    parser.add_argument("--df_eval_sampling_seed", type=int, default=1234,
                        help="fixed DF subset seed, independent of the training seed")
    parser.add_argument("--la_official_eval", dest="la_official_eval", action="store_true", default=True)
    parser.add_argument("--no_la_official_eval", dest="la_official_eval", action="store_false")
    parser.add_argument("--la_eval_phase", type=str, default="eval", choices=["progress", "eval", "hidden_track"])
    parser.add_argument("--la_eval_sampling_ratio", type=float, default=0.3)
    parser.add_argument("--la_eval_sampling_seed", type=int, default=1234)
    parser.add_argument("--show_sample_examples", type=int, default=5)
    parser.add_argument("--eval_len", type=int, default=64600)
    parser.add_argument("--mlf_weight_batches", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.itw_only_val:
        raise ValueError(
            "This entry point is intentionally ITW-only. Add --itw_only_val, "
            "or use the ITW trainer for DF/LA validation."
        )
    # ITW-only implies ITW validation and best-ITW checkpoint selection.
    args.use_itw_val = True
    args.itw_monitor_only = False

    set_seed(args.seed, deterministic=args.deterministic)
    out_dir = Path(args.out_fold)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model = build_model(args, device)
    load_initial_checkpoint(model, args.init_ckpt, device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(args.amp and device == "cuda" and args.amp_dtype == "fp16")
    )
    print(
        f"[speed] amp={args.amp and device == 'cuda'} dtype={args.amp_dtype} "
        f"itw_eval_batch_size={args.itw_eval_batch_size} tf32={args.tf32}"
    )

    # Train only on ASVspoof2019 LA train.
    db = Path(args.data_2019la).expanduser()
    train_proto = db / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    d_label, file_train = genSpoof_list(train_proto, is_train=True)
    excluded = load_exclude_utts(args.exclude_utts)
    file_train = [u for u in file_train if u not in excluded]
    train_set = Dataset_2019LA_train(
        file_train,
        d_label,
        db / "ASVspoof2019_LA_train",
        rawboost_algos=parse_int_list_csv(args.rawboost_algos),
        name="2019LA-train",
        on_error=args.audio_error,
    )
    train_loader = make_loader(train_set, args, train=True, shuffle=True, drop_last=True)

    if args.itw_eval_interval < 1:
        raise ValueError("--itw_eval_interval must be >= 1")
    if args.itw_num_crops < 1:
        raise ValueError("--itw_num_crops must be >= 1")
    if args.itw_eval_batch_size < 1:
        raise ValueError("--itw_eval_batch_size must be >= 1")
    itw_loader, itw_labels, itw_protocol = build_itw_eval(args)

    print(f"Train: 2019LA train N={len(train_set)}")
    print(
        f"Validate/select ONLY ITW: N={len(itw_loader.dataset)} "
        f"interval={args.itw_eval_interval} crops={args.itw_num_crops} "
        f"batch={args.itw_eval_batch_size} protocol={itw_protocol}"
    )
    print(
        "[ITW][NOTICE] Because ITW selects checkpoints in this run, its best score is a "
        "validation result rather than an untouched final test result."
    )
    print(f"Score convention: score_mode={args.score_mode}; larger score = more bonafide")

    optimizer = build_optimizer(model, args)
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_lambda(step):
        min_ratio = min(1.0, max(0.0, float(args.min_lr_ratio)))
        warm_start = min(1.0, max(0.0, float(args.warmup_start_ratio)))
        if step < args.warmup:
            frac = step / max(1, args.warmup)
            return warm_start + (1.0 - warm_start) * frac
        progress = (step - args.warmup) / max(1, total_steps - args.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    steps_per_epoch = max(1, len(train_loader))
    print(
        f"[sched] steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"warmup={args.warmup} steps (~{args.warmup / steps_per_epoch:.1f} epochs)"
    )
    if args.warmup > 0.25 * total_steps:
        print("[sched][WARN] warmup covers >25% of training; consider lowering --warmup.")

    log_csv = out_dir / "train_log.csv"
    HEADER = ("epoch,train_loss,train_eer,itw_loss,itw_acc,itw_eer,itw_eer_inv,"
              "itw_separation,itw_n_scored,itw_best,time_s\n")

    resume_epoch = args.resume_from_epoch if args.resume_dir else 0

    if resume_epoch >= 1:
        # ── continue an interrupted run (operational; recipe/LR/monitor unchanged) ──
        rdir = Path(args.resume_dir)
        rck = rdir / "checkpoint" / f"epoch_{resume_epoch:03d}.pt"
        if not rck.is_file():
            raise FileNotFoundError(f"--resume_dir checkpoint missing: {rck}")
        state = torch.load(rck, map_location=device, weights_only=False)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if state and all(str(k).startswith("module.") for k in state):
            state = {str(k)[7:]: v for k, v in state.items()}
        mm = model.module if hasattr(model, "module") else model
        inc = mm.load_state_dict(state, strict=True)
        if inc.missing_keys or inc.unexpected_keys:
            raise RuntimeError(f"resume ckpt mismatch: missing={inc.missing_keys} unexpected={inc.unexpected_keys}")
        # best-ITW history from existing train_log.csv (rows for completed epochs)
        best_itw_eer = float("inf")
        best_itw_epoch = -1
        prev_rows = []
        if log_csv.is_file():
            with log_csv.open("r", encoding="utf-8", errors="ignore") as f:
                prev_rows = list(csv.DictReader(f))
        for r in prev_rows:
            try:
                ep = int(float(r["epoch"]))
                v = float(r["itw_eer"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isnan(v):
                continue
            if v < best_itw_eer:
                best_itw_eer, best_itw_epoch = v, ep
        if best_itw_epoch < 0:
            raise RuntimeError(f"resume: no usable eval rows in {log_csv} (need completed epochs)")
        # warm-start cosine LR to where the interrupted run was
        for _ in range(resume_epoch * steps_per_epoch):
            scheduler.step()
        start_epoch = resume_epoch + 1
        print(f"[resume] loaded {rck} | continuing epochs {start_epoch}..{args.epochs} | "
              f"warm-started scheduler to step={resume_epoch * steps_per_epoch} | "
              f"history: itw_best={best_itw_eer:.3f}@ep{best_itw_epoch}")
    else:
        best_itw_eer = float("inf")
        best_itw_epoch = -1
        start_epoch = 1

    if resume_epoch >= 1 and log_csv.is_file():
        pass  # append rows below; do NOT rewrite header (keeps completed epochs)
    else:
        with log_csv.open("w", encoding="utf-8") as f:
            f.write(HEADER)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_eer = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            best_itw_eer,
            args,
            scaler=scaler,
        )

        ran_itw = epoch % int(args.itw_eval_interval) == 0 or epoch == int(args.epochs)
        if ran_itw:
            itw_loss, itw_acc, itw_eer, _, itw_stats = evaluate_itw(
                model,
                itw_loader,
                device,
                itw_labels,
                desc="ITW-validation",
                args=args,
            )
            is_best_itw = itw_eer < best_itw_eer
            if is_best_itw:
                best_itw_eer = float(itw_eer)
                best_itw_epoch = int(epoch)
                torch.save(model.state_dict(), out_dir / "best_itw_model.pt")
                torch.save(optimizer.state_dict(), out_dir / "best_itw_op.pt")
        else:
            itw_loss = itw_acc = itw_eer = float("nan")
            itw_stats = {
                "eer_inv": float("nan"),
                "separation": float("nan"),
                "n_scored": 0,
            }
            is_best_itw = False

        torch.save(model.state_dict(), out_dir / "last_model.pt")
        if args.save_epoch_ckpts:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        elapsed = time.time() - t0
        print(f"  train loss={train_loss:.4f} EER={train_eer:.2f}% | time={elapsed:.0f}s")
        if ran_itw:
            print(
                f"  ITW EER={itw_eer:.2f}% inv={itw_stats['eer_inv']:.2f}% "
                f"sep={itw_stats['separation']:.3f} N={itw_stats['n_scored']}"
                f"{' ★ best-itw' if is_best_itw else ''}"
            )

        with log_csv.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_eer:.4f},"
                f"{itw_loss:.6f},{itw_acc:.4f},{itw_eer:.4f},"
                f"{itw_stats['eer_inv']:.4f},{itw_stats['separation']:.6f},"
                f"{int(itw_stats['n_scored'])},{'yes' if is_best_itw else ''},{elapsed:.0f}\n"
            )

    best_itw_path = out_dir / "best_itw_model.pt"
    if not best_itw_path.is_file():
        raise RuntimeError("No best_itw_model.pt was produced; check ITW protocol/audio loading")

    model.load_state_dict(torch.load(best_itw_path, map_location=device))
    print(
        f"\nLoaded best_itw_model.pt from epoch {best_itw_epoch}, "
        f"recorded best ITW EER={best_itw_eer:.2f}%"
    )
    _, _, final_itw_eer, _, final_itw_stats = evaluate_itw(
        model,
        itw_loader,
        device,
        itw_labels,
        desc="ITW-validation-final-best-itw",
        args=args,
        return_scores=True,
    )
    produce_itw_scores(final_itw_stats, out_dir / "ITW_scores.best_itw.txt")

    with (out_dir / "final_validation_result.txt").open("w", encoding="utf-8") as f:
        f.write("selection_protocol=itw_only_validation\n")
        f.write(f"itw_protocol={itw_protocol}\n")
        f.write(f"itw_sample_ratio={args.itw_sample_ratio}\n")
        f.write(f"itw_sampling_seed={args.itw_sampling_seed}\n")
        f.write(f"itw_num_crops={args.itw_num_crops}\n")
        f.write(f"best_itw_epoch={best_itw_epoch}\n")
        f.write(f"best_itw_eer_recorded={best_itw_eer:.6f}\n")
        f.write(f"itw_eer_final_best_itw={final_itw_eer:.6f}\n")
        f.write(f"itw_n_scored={final_itw_stats['n_scored']}\n")
        f.write(f"itw_n_bonafide={final_itw_stats['n_bonafide']}\n")
        f.write(f"itw_n_spoof={final_itw_stats['n_spoof']}\n")
        f.write(f"itw_n_unreadable={final_itw_stats['n_unreadable']}\n")

    print(
        f"\n[FINAL][ITW-validation][best_itw_model] EER={final_itw_eer:.2f}% "
        f"sep={final_itw_stats['separation']:.3f} N={final_itw_stats['n_scored']}"
    )


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = int(time.time() - start)
    print(f"Total time: {elapsed // 3600}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
