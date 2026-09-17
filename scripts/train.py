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
    rank = pairwise_rank_loss(logits, labels, margin=args.rank_margin, max_pairs=args.rank_max_pairs)
    lam_rank = aux_scale(epoch, args.aux_start_epoch, args.aux_warmup_epochs, args.rank_lambda)
    lam_oc = aux_scale(epoch, args.aux_start_epoch, args.aux_warmup_epochs, args.oc_lambda)
    if not hasattr(model, "oc_head"):
        raise RuntimeError("Current best loss requires model.oc_head. Keep --use_oc enabled.")
    loss = wce + lam_rank * rank + lam_oc * model.oc_head(emb, labels)
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


def sample_eval_keys_class_ratio(keys, labels, sample_ratio=1.0, bon_ratio=1.0, seed=1234):
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


def _run_eval_robust(model, loader, labels, attacks, desc, args, device, epoch, max_tries=4):
    """evaluate() + DataLoader-worker-death retry (operational resilience only).

    If a DataLoader worker is SIGKILLed mid-eval (observed as the container OOM-killer taking
    out a worker, which surfaces in torch as "worker ... killed by signal: Killed"), the eval
    loader is rebuilt from the same dataset and the pass is retried. Eval loaders are
    shuffle=False / no-dropout / no-grad, so a retried pass is numerically identical to the
    interrupted one. The rebuild keeps the exact loader config via make_loader(..., train=False).
    """
    import gc

    def _rebuild():
        ds = loader.dataset
        return make_loader(ds, args, train=False)

    for attempt in range(1, max_tries + 1):
        try:
            res = evaluate(model, loader, device, labels, attacks, desc=desc, args=args, epoch=epoch)
            return res, loader
        except RuntimeError as e:
            msg = str(e).lower()
            worker_dead = ("worker" in msg) and ("killed" in msg or "fail" in msg)
            if not worker_dead:
                raise
            if attempt == max_tries:
                raise RuntimeError(f"{desc}: DataLoader worker died {max_tries}x; giving up (epoch {epoch})")
            print(f"[robust] {desc}: DataLoader worker died (attempt {attempt}/{max_tries}); "
                  f"rebuilding shuffle=False eval loader and retrying epoch {epoch}", flush=True)
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loader = _rebuild()
    raise RuntimeError("unreachable")  # pragma: no cover


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
    # Released model: k-query tokenizer, attn cross-level fusion, align=middle,
    # prior strength 0, preact fix, local-global alternating Mamba-2 blocks.
    d_args = {
        "d_model": args.d_model,
        "d_state": args.d_state,
        "n_layer": args.n_layer,
        "num_classes": 2,
        "n_query": args.n_query,
        "sinc_channels": args.sinc_channels,
        "freq_pool": args.freq_pool,
        "pool_heads": args.pool_heads,
        "headdim": args.headdim,
        "dropout": args.dropout,
        "use_oc": True,          # OC-Softmax is an auxiliary loss only
        "preact_fix": args.preact_fix,
    }
    model = Model(d_args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: FALG-Mamba (released) params={n_params/1e6:.3f}M device={device}")
    print("[score] raw classifier logits (logits[:,1]-logits[:,0]); OC-Softmax = auxiliary loss only")
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


def _build_2019la_full(args, partition, excluded=None):
    """Full official 2019LA partition loader (no subsampling)."""
    root = Path(args.data_2019la)
    proto = root / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{partition}.trl.txt"
    labels, utts = genSpoof_list(proto, is_train=True)
    if excluded:
        utts = [u for u in utts if u not in excluded]
        labels = {k: v for k, v in labels.items() if k in set(utts)}
    ds = Dataset_Audio_eval(utts, root / f"ASVspoof2019_LA_{partition}", ext=".flac",
                            target_len=args.eval_len, name=f"2019LA-{partition}", on_error=args.audio_error)
    return make_loader(ds, args, train=False), labels, {u: "-" for u in utts}


def build_2019la_dev(args, excluded=None):
    return _build_2019la_full(args, "dev", excluded)


def build_2019la_eval(args, excluded=None):
    return _build_2019la_full(args, "eval", excluded)


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
    parser.add_argument("--route", type=str, default="dfla", choices=["la19", "dfla"],
                        help="la19: test on ASVspoof2019 LA eval; dfla: test on 2021DF + 2021LA (full official partitions)")
    parser.add_argument("--data_2019la", type=str, default="/root/autodl-tmp/data/2019LA")
    parser.add_argument("--data_2021df_flac", type=str, default="/root/autodl-tmp/data/2021DF/ASVspoof2021_DF_eval/flac")
    parser.add_argument("--trial_metadata", type=str, default="/root/autodl-tmp/data/2021DF/keys/CM/trial_metadata.txt")
    parser.add_argument("--data_2021la_flac", type=str, default="/root/autodl-tmp/data/2021LA/ASVspoof2021_LA_eval/flac")
    parser.add_argument("--data_2021la_meta", type=str, default="/root/autodl-tmp/data/2021LA/keys/CM/trial_metadata.txt")
    parser.add_argument("--data_2021la_keys_dir", type=str, default="")

    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_query", type=int, default=4)
    parser.add_argument("--sinc_channels", type=int, default=70)
    parser.add_argument("--pool_heads", type=int, default=4)
    parser.add_argument("--headdim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freq_pool", type=int, default=3)
    # Cross-level fusion: which level's time grid the other levels are aligned onto.
    # The three levels have very different lengths (~597 / 99 / 16 frames at 64600 samples),
    # so 'deep' collapses the shallow level ~37x. 'middle' is the new default.
    # Pre-activation fix in the residual blocks (legacy RawNet2 baseline discards it).
    parser.add_argument("--preact_fix", dest="preact_fix", action="store_true", default=True)
    parser.add_argument("--no_preact_fix", dest="preact_fix", action="store_false")

    parser.add_argument("--w_spoof", type=float, default=0.1)
    parser.add_argument("--w_bonafide", type=float, default=0.9)
    parser.add_argument("--rank_lambda", type=float, default=0.1)
    parser.add_argument("--rank_margin", type=float, default=0.5)
    parser.add_argument("--rank_max_pairs", type=int, default=4096)
    parser.add_argument("--oc_lambda", type=float, default=0.02)
    parser.add_argument("--aux_start_epoch", type=int, default=8)
    parser.add_argument("--aux_warmup_epochs", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base_lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=3000)
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

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default="",
                        help="resume an interrupted run: dir whose checkpoint/epoch_<N>.pt continues; "
                             "requires --resume_from_epoch>=1. Loads weights, warm-starts cosine LR to "
                             "that epoch's step, keeps best-monitor history from train_log.csv, appends rows. "
                             "Does NOT reset optimizer adaptive state (only Adam's m/v restart).")
    parser.add_argument("--resume_from_epoch", type=int, default=0,
                        help="epoch already completed when --resume_dir set; training continues at this+1")
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

    parser.add_argument("--rawboost_algos", type=str, default="1,2,3")
    # Only the official 2021DF 'eval' partition is used for validation by default.
    parser.add_argument("--df_eval_phase", type=str, default="eval",
                        choices=["progress", "eval", "hidden_track", "all"])
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--bon_ratio", type=float, default=1.0)
    parser.add_argument("--df_eval_sampling_seed", type=int, default=1234,
                        help="fixed DF subset seed, independent of the training seed")
    parser.add_argument("--la_official_eval", dest="la_official_eval", action="store_true", default=True)
    parser.add_argument("--no_la_official_eval", dest="la_official_eval", action="store_false")
    parser.add_argument("--la_eval_phase", type=str, default="eval", choices=["progress", "eval", "hidden_track"])
    parser.add_argument("--la_eval_sampling_ratio", type=float, default=1.0)
    parser.add_argument("--la_eval_sampling_seed", type=int, default=1234)
    parser.add_argument("--show_sample_examples", type=int, default=5)
    parser.add_argument("--eval_len", type=int, default=64600)
    parser.add_argument("--mlf_weight_batches", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    Path(args.out_fold).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.out_fold) / "checkpoint"
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
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda" and args.amp_dtype == "fp16"))
    print(f"[speed] amp={args.amp and device == 'cuda'} dtype={args.amp_dtype} eval_batch_size={args.eval_batch_size} tf32={args.tf32}")

    db = Path(args.data_2019la)
    train_proto = db / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    d_label, file_train = genSpoof_list(train_proto, is_train=True)
    excluded = load_exclude_utts(args.exclude_utts)
    file_train = [u for u in file_train if u not in excluded]
    train_set = Dataset_2019LA_train(file_train, d_label, db / "ASVspoof2019_LA_train", rawboost_algos=parse_int_list_csv(args.rawboost_algos), name="2019LA-train", on_error=args.audio_error)
    train_loader = make_loader(train_set, args, train=True, shuffle=True, drop_last=True)

    # Checkpoint selection uses the FULL official ASVspoof2019 LA dev partition.
    dev_loader, dev_labels, dev_attacks = build_2019la_dev(args, excluded=excluded)

    print(f"Train: 2019LA train N={len(train_set)}")
    print(f"Validate/select: 2019LA official dev N={len(dev_loader.dataset)} -> lowest dev EER")
    print(f"Test (full official partitions): 2021DF phase={args.df_eval_phase} | 2021LA phase={args.la_eval_phase}")
    print(f"Fixed recipe: rawboost={args.rawboost_algos} loss=WCE+Rank+OC({args.oc_lambda}) fusion=attn prior=0.0")
    print("Score convention: raw un-normalized logits[:,1]-logits[:,0]")

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
    print(f"[sched] steps/epoch={steps_per_epoch} total_steps={total_steps} warmup={args.warmup} steps")

    log_csv = Path(args.out_fold) / "train_log.csv"
    mlf_csv = Path(args.out_fold) / "mlf_weight_log.csv"
    HEADER = "epoch,train_loss,train_eer,dev_loss,dev_acc,dev_eer,dev_eer_inv,dev_separation,dev_best,time_s\n"

    resume_epoch = args.resume_from_epoch if args.resume_dir else 0
    if resume_epoch >= 1:
        rck = Path(args.resume_dir) / "checkpoint" / f"epoch_{resume_epoch:03d}.pt"
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
            raise RuntimeError(f"resume mismatch: missing={inc.missing_keys} unexpected={inc.unexpected_keys}")
        best_dev_eer, best_dev_epoch = float("inf"), -1
        if log_csv.is_file():
            for r in csv.DictReader(log_csv.open("r", encoding="utf-8", errors="ignore")):
                try:
                    if float(r["dev_eer"]) < best_dev_eer:
                        best_dev_eer, best_dev_epoch = float(r["dev_eer"]), int(float(r["epoch"]))
                except (KeyError, TypeError, ValueError):
                    continue
        for _ in range(resume_epoch * steps_per_epoch):
            scheduler.step()
        start_epoch = resume_epoch + 1
        print(f"[resume] {rck} -> epochs {start_epoch}..{args.epochs} | dev_best={best_dev_eer:.3f}@ep{best_dev_epoch}")
    else:
        best_dev_eer, best_dev_epoch, start_epoch = float("inf"), -1, 1

    if resume_epoch >= 1 and log_csv.is_file():
        pass
    else:
        with log_csv.open("w", encoding="utf-8") as f:
            f.write(HEADER)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, train_eer = train_epoch(model, train_loader, optimizer, scheduler, device,
                                            epoch, best_dev_eer, args, scaler=scaler)
        (dev_loss, dev_acc, dev_eer, dev_pa, dev_stats), dev_loader = _run_eval_robust(
            model, dev_loader, dev_labels, dev_attacks, "2019LA-dev", args, device, epoch)

        is_best = dev_eer < best_dev_eer
        if is_best:
            best_dev_eer, best_dev_epoch = dev_eer, epoch
            torch.save(model.state_dict(), Path(args.out_fold) / "best_model.pt")
            torch.save(optimizer.state_dict(), Path(args.out_fold) / "best_model_op.pt")
        torch.save(model.state_dict(), Path(args.out_fold) / "last_model.pt")
        if args.save_epoch_ckpts:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        mlf_stats = collect_mlf_weight_stats(model, dev_loader, device, max_batches=args.mlf_weight_batches)
        mlf_msg = format_mlf_stats(mlf_stats)
        if mlf_msg:
            print(f"  [MLF] {mlf_msg}")
            write_mlf_stats_csv(str(mlf_csv), epoch, mlf_stats)

        elapsed = time.time() - t0
        print(f"  train loss={train_loss:.4f} EER={train_eer:.2f}% | "
              f"dev EER={dev_eer:.2f}% inv={dev_stats['eer_inv']:.2f}% sep={dev_stats['separation']:.3f}"
              f"{' <-- best' if is_best else ''} | time={elapsed:.0f}s")
        with log_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_eer:.4f},"
                    f"{dev_loss:.6f},{dev_acc:.4f},{dev_eer:.4f},{dev_stats['eer_inv']:.4f},"
                    f"{dev_stats['separation']:.6f},{'yes' if is_best else ''},{elapsed:.0f}\n")

    # ── Final: evaluate the dev-selected checkpoint on the FULL official test partitions ──
    model.load_state_dict(torch.load(Path(args.out_fold) / "best_model.pt", map_location=device))
    print(f"\nLoaded best_model.pt from epoch {best_dev_epoch} (dev EER={best_dev_eer:.2f}%)")

    if args.route == "la19":
        ev_loader, ev_labels, ev_attacks = build_2019la_eval(args, excluded=excluded)
        ev_loss, ev_acc, ev_eer, ev_pa, ev_stats = evaluate(
            model, ev_loader, device, ev_labels, ev_attacks, desc="2019LA-eval-full", args=args, epoch=best_dev_epoch)
        produce_scores(model, ev_loader, device, Path(args.out_fold) / "2019LA_eval_scores.best.txt", args=args)
        with (Path(args.out_fold) / "final_result.txt").open("w", encoding="utf-8") as f:
            f.write("model=FALG-Mamba\nroute=la19\nselection=2019LA-official-dev-lowest-EER\n")
            f.write(f"rawboost_algos={args.rawboost_algos}\nloss=wce_rank_oc\noc_lambda={args.oc_lambda}\n")
            f.write(f"best_epoch={best_dev_epoch}\nbest_dev_eer={best_dev_eer:.6f}\n")
            f.write(f"2019la_eer_full={ev_eer:.6f}\n")
        print(f"\n[FINAL][2019LA-eval-full] EER={ev_eer:.2f}% sep={ev_stats['separation']:.3f}")
    else:
        df_loader, df_labels, df_attacks = build_2021df_eval(args, excluded=excluded)
        la_loader, la_labels, la_attacks = build_2021la_eval(args, excluded=excluded)
        df_loss, df_acc, df_eer, df_pa, df_stats = evaluate(
            model, df_loader, device, df_labels, df_attacks, desc="2021DF-eval-full", args=args, epoch=best_dev_epoch)
        produce_scores(model, df_loader, device, Path(args.out_fold) / "2021DF_eval_scores.best.txt", args=args)
        la_loss, la_acc, la_eer, la_pa, la_stats = evaluate(
            model, la_loader, device, la_labels, la_attacks, desc="2021LA-eval-full", args=args, epoch=best_dev_epoch)
        produce_scores(model, la_loader, device, Path(args.out_fold) / "2021LA_eval_scores.best.txt", args=args)
        with (Path(args.out_fold) / "final_result.txt").open("w", encoding="utf-8") as f:
            f.write("model=FALG-Mamba\nroute=dfla\nselection=2019LA-official-dev-lowest-EER\n")
            f.write(f"rawboost_algos={args.rawboost_algos}\nloss=wce_rank_oc\noc_lambda={args.oc_lambda}\n")
            f.write(f"best_epoch={best_dev_epoch}\nbest_dev_eer={best_dev_eer:.6f}\n")
            f.write(f"2021df_eer_full={df_eer:.6f}\n2021la_eer_full={la_eer:.6f}\n")
            if la_stats.get("official_eer") is not None:
                f.write(f"2021la_official_eer={la_stats['official_eer']:.6f}\n")
                if la_stats.get("min_tDCF") is not None:
                    f.write(f"2021la_min_tDCF={la_stats['min_tDCF']:.6f}\n")
        print(f"\n[FINAL][2021DF-eval-full] EER={df_eer:.2f}% sep={df_stats['separation']:.3f}")
        print(f"[FINAL][2021LA-eval-full] EER={la_eer:.2f}% sep={la_stats['separation']:.3f}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = int(time.time() - start)
    print(f"Total time: {elapsed // 3600}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
