#!/usr/bin/env python3
"""
Full-partition evaluation for FALG-Mamba-MLF-PG.

What this fixes relative to the training loop
---------------------------------------------
The trainer validates on a *sub-sample* (20% of 2021DF, 30% attack-balanced of
2021LA) and saves two different checkpoints (best_df_model.pt / best_la_model.pt).
Neither is a number you can put in a paper:

  * the EER is computed on a subset, not the official partition;
  * DF and LA are reported from *different* models.

This script scores ONE model on the FULL official `eval` partition of both sets
(and optionally In-the-Wild), writes official-format score files, and reports the
official pooled EER (+ min t-DCF for LA).  Run it once per checkpoint you want to
report, and put a single row in the paper.

It also supports checkpoint averaging (`--ckpt a.pt b.pt c.pt`), which is what
RawTFNet and most ESPnet-style recipes do to damp the epoch-to-epoch EER jitter.
NOTE: this only helps if you actually SAVED per-epoch checkpoints. The current
trainer only keeps best_df/best_la, so for the existing run you can average at
most those two.

Usage
-----
# single checkpoint, both sets, full eval partitions
python scripts/eval_full.py \
  --ckpt ./exp/best_df_model.pt \
  --tag best_df \
  --sets df,la \
  --out_dir ./eval_full

# checkpoint averaging (future runs, once you keep per-epoch checkpoints)
python scripts/eval_full.py \
  --ckpt ./exp/checkpoint/ep_37.pt ./exp/checkpoint/ep_39.pt ./exp/checkpoint/ep_42.pt \
  --tag avg_top3 --sets df,la,itw --itw_dir /root/autodl-tmp/data/in_the_wild

# smoke test on 2000 trials before committing to the full 500k
python scripts/eval_full.py --ckpt ./exp/best_df_model.pt --tag smoke --sets df --limit 2000

Score convention (must match training)
--------------------------------------
    spoof = 0, bonafide = 1
    score = logits[:, 1] - logits[:, 0]   (larger = more bonafide)
    --score_mode raw  -> raw un-normalised classifier logits (training default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils_eval_extra import BAD_AUDIO_SENTINEL, Dataset_Audio_eval
from model_scripts.AA_LG_Mamba_MLF_PG import Model

import eval_metric_LA as em


# ─────────────────────────────────────────────────────────────────────────────
# Metadata: FULL official partition, no sampling
# ─────────────────────────────────────────────────────────────────────────────
# ASVspoof2021 keys/CM/trial_metadata.txt column layout (LA and DF alike):
#   0 speaker | 1 utt | 2 codec/compression | 3 source | 4 attack | 5 label | 6 trim | 7 phase
LABEL_COL, PHASE_COL = 5, 7
DEFAULT_CODEC_COL, DEFAULT_ATTACK_COL = 2, 4
PHASES = ("progress", "eval", "hidden_track")


class Trials:
    """Container for one evaluation partition."""

    def __init__(self, name: str):
        self.name = name
        self.keys: List[str] = []
        self.label: Dict[str, int] = {}
        self.codec: Dict[str, str] = {}
        self.attack: Dict[str, str] = {}

    def __len__(self):
        return len(self.keys)

    def counts(self):
        b = sum(1 for k in self.keys if self.label[k] == 1)
        return b, len(self.keys) - b

    def drop(self, excluded: set):
        if not excluded:
            return 0
        before = len(self.keys)
        self.keys = [k for k in self.keys if k not in excluded]
        return before - len(self.keys)


def load_asvspoof_trials(meta_path, phase: str, name: str,
                         codec_col: int = DEFAULT_CODEC_COL,
                         attack_col: int = DEFAULT_ATTACK_COL) -> Trials:
    """Read the FULL trial list of one official partition. No sub-sampling."""
    meta = Path(meta_path)
    if not meta.exists():
        raise FileNotFoundError(f"[{name}] metadata not found: {meta}")
    if phase not in PHASES + ("all",):
        raise ValueError(f"phase must be one of {PHASES + ('all',)}, got {phase!r}")

    t = Trials(name)
    phase_counts: Counter = Counter()
    n_rows = 0
    seen = set()
    with meta.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) <= PHASE_COL:
                continue
            lab_tok = parts[LABEL_COL].lower()
            if lab_tok not in ("bonafide", "spoof"):
                continue
            n_rows += 1
            row_phase = parts[PHASE_COL].lower()
            phase_counts[row_phase] += 1
            if phase != "all" and row_phase != phase:
                continue
            utt = parts[1]
            if utt in seen:
                continue
            seen.add(utt)
            t.keys.append(utt)
            t.label[utt] = 1 if lab_tok == "bonafide" else 0
            t.codec[utt] = parts[codec_col] if len(parts) > codec_col else "-"
            t.attack[utt] = "bonafide" if lab_tok == "bonafide" else (
                parts[attack_col] if len(parts) > attack_col else "-")

    dist = "  ".join(f"{k}={v}" for k, v in sorted(phase_counts.items()))
    print(f"[{name}] {meta}")
    print(f"[{name}] labelled rows={n_rows}   phase distribution: {dist}")
    if not t.keys:
        raise RuntimeError(f"[{name}] phase='{phase}' selected 0 trials. Available: {dist or 'none'}")
    b, s = t.counts()
    print(f"[{name}] phase='{phase}' -> N={len(t):,d}  bonafide={b:,d}  spoof={s:,d}   (FULL partition, no sampling)")
    print(f"[{name}] codecs found (col {codec_col}): {sorted(set(t.codec.values()))}")
    return t


def load_itw_trials(meta_path, name: str = "ITW") -> Trials:
    """In-the-Wild meta.csv:  file,speaker,label  with label in {bona-fide, spoof}."""
    meta = Path(meta_path)
    if not meta.exists():
        raise FileNotFoundError(f"[{name}] meta.csv not found: {meta}")
    import csv as _csv
    t = Trials(name)
    with meta.open("r", encoding="utf-8", errors="ignore") as f:
        for row in _csv.DictReader(f):
            utt = Path(str(row["file"]).strip()).stem
            lab = 1 if str(row["label"]).strip().lower().replace("_", "-") in ("bona-fide", "bonafide") else 0
            t.keys.append(utt)
            t.label[utt] = lab
            t.codec[utt] = "-"
            t.attack[utt] = "bonafide" if lab == 1 else "spoof"
    b, s = t.counts()
    print(f"[{name}] N={len(t):,d}  bonafide={b:,d}  spoof={s:,d}")
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
def load_and_maybe_average(paths: List[str]) -> Tuple[dict, dict]:
    """Load N state_dicts and average the float tensors (checkpoint averaging).

    Integer buffers (num_batches_tracked) are taken from the first checkpoint.
    BatchNorm running stats ARE averaged, which is what ESPnet/RawTFNet do.
    """
    sds = []
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sds.append(sd)

    ref = sds[0]
    for i, sd in enumerate(sds[1:], 1):
        if set(sd.keys()) != set(ref.keys()):
            raise RuntimeError(f"checkpoint {paths[i]} has different keys than {paths[0]}")

    if len(sds) == 1:
        return ref, {"n_ckpt": 1, "ckpts": [str(p) for p in paths]}

    avg = {}
    for k, v0 in ref.items():
        if torch.is_floating_point(v0):
            avg[k] = torch.stack([sd[k].float() for sd in sds], 0).mean(0).to(v0.dtype)
        else:
            avg[k] = v0.clone()
    print(f"[ckpt] averaged {len(sds)} checkpoints (float tensors incl. BN running stats)")
    return avg, {"n_ckpt": len(sds), "ckpts": [str(p) for p in paths]}


def build_model(args, device):
    d_args = {
        "d_model": args.d_model, "d_state": args.d_state, "n_layer": args.n_layer,
        "num_classes": 2, "n_query": args.n_query, "scan": args.scan,
        "use_oc": True,
        "logit_source": "cls" if args.score_mode == "raw" else "oc",
        "oc_alpha": args.oc_alpha, "oc_m_real": args.oc_m_real, "oc_m_fake": args.oc_m_fake,
        "mlf_fusion": args.mlf_fusion, "mlf_prior_init": args.mlf_prior_init,
        "mlf_prior_strength": args.mlf_prior_strength, "mlf_align_level": args.mlf_align_level,
        "sinc_channels": args.sinc_channels, "freq_pool": args.freq_pool,
        "pool_heads": args.pool_heads, "headdim": args.headdim, "dropout": args.dropout,
        "use_gate": True, "freq_flatten": args.freq_flatten, "flatten_mode": args.flatten_mode,
        "preact_fix": args.preact_fix,
        # Exp B switch: fixed_band adds no learnable params; k_query default unchanged.
        "freq_tokenizer": getattr(args, "freq_tokenizer", "k_query"),
    }
    model = Model(d_args).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n/1e6:.3f}M  align_level={args.mlf_align_level}  fusion={args.mlf_fusion}  "
          f"score_mode={args.score_mode}  device={device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_partition(model, trials: Trials, audio_dir, ext, args, device):
    ds = Dataset_Audio_eval(trials.keys, str(audio_dir), ext=ext, target_len=args.eval_len,
                            name=trials.name, verbose=True, on_error="skip")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=2 if args.num_workers > 0 else None)
    model.eval()

    amp = (not args.fp32) and device.startswith("cuda")
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True) if amp \
        else torch.autocast(device_type="cpu", enabled=False)

    keys_out, scores_out, bad = [], [], []
    t0 = time.time()
    for data, keys in tqdm(loader, desc=f"score {trials.name}", ncols=110):
        keys = [str(k) for k in keys]
        keep = [i for i, k in enumerate(keys) if not k.startswith(BAD_AUDIO_SENTINEL)]
        bad.extend(k[len(BAD_AUDIO_SENTINEL):] for k in keys if k.startswith(BAD_AUDIO_SENTINEL))
        if not keep:
            continue
        data = data.to(device, non_blocking=True)
        with ctx:
            out = model(data)
            logits = out[0] if isinstance(out, tuple) else out
        logits = logits.float()
        s = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        for i in keep:
            keys_out.append(keys[i])
            scores_out.append(float(s[i]))

    dt = time.time() - t0
    print(f"[{trials.name}] scored {len(keys_out):,d} trials in {dt/60:.1f} min "
          f"({len(keys_out)/max(dt,1e-9):.0f} utt/s); undecodable={len(bad)}")
    return keys_out, np.asarray(scores_out, dtype=float), sorted(set(bad))


def eer_of(labels: np.ndarray, scores: np.ndarray) -> float:
    tgt = scores[labels == 1]
    non = scores[labels == 0]
    if len(tgt) == 0 or len(non) == 0:
        return float("nan")
    return float(em.compute_eer(tgt, non)[0]) * 100.0


def breakdown(keys, scores, trials: Trials, group: Dict[str, str], min_n=30):
    """EER of every group's spoof trials against the pooled bonafide trials."""
    labels = np.array([trials.label[k] for k in keys], dtype=int)
    bona = scores[labels == 1]
    rows = []
    by_group = defaultdict(list)
    for k, s in zip(keys, scores):
        if trials.label[k] == 0:
            by_group[group.get(k, "-")].append(s)
    for g, ss in sorted(by_group.items()):
        if len(ss) < min_n:
            continue
        lab = np.concatenate([np.ones(len(bona), int), np.zeros(len(ss), int)])
        sc = np.concatenate([bona, np.asarray(ss, float)])
        rows.append((g, len(ss), eer_of(lab, sc)))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser("Full-partition evaluation")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one checkpoint, or several to average (checkpoint averaging)")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out_dir", default="./eval_full")
    ap.add_argument("--sets", default="df,la", help="comma list of df,la,itw")

    # data
    ap.add_argument("--data_2021df_flac", default="/root/autodl-tmp/data/2021DF/ASVspoof2021_DF_eval/flac")
    ap.add_argument("--data_2021df_meta", default="/root/autodl-tmp/data/2021DF/keys/CM/trial_metadata.txt")
    ap.add_argument("--data_2021la_flac", default="/root/autodl-tmp/data/2021LA/ASVspoof2021_LA_eval/flac")
    ap.add_argument("--data_2021la_meta", default="/root/autodl-tmp/data/2021LA/keys/CM/trial_metadata.txt")
    ap.add_argument("--data_2021la_keys_dir", default="")
    ap.add_argument("--itw_dir", default="")
    ap.add_argument("--itw_meta", default="", help="default: <itw_dir>/meta.csv")
    ap.add_argument("--df_eval_phase", default="eval", choices=list(PHASES) + ["all"])
    ap.add_argument("--la_eval_phase", default="eval", choices=list(PHASES) + ["all"])
    ap.add_argument("--exclude_utts", default="", help="e.g. output of scripts/scan_audio.py")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first N trials")

    # model (defaults MUST match the training run)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_query", type=int, default=4)
    ap.add_argument("--scan", default="alt", choices=["alt", "uni", "bi"])
    ap.add_argument("--sinc_channels", type=int, default=70)
    ap.add_argument("--pool_heads", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freq_pool", type=int, default=3)
    ap.add_argument("--mlf_align_level", default="middle", choices=["shallow", "middle", "deep"])
    ap.add_argument("--mlf_fusion", default="attn", choices=["attn", "prior_attn", "prior_only", "mean"])
    ap.add_argument("--mlf_prior_init", default="0.33,0.33,0.34")
    ap.add_argument("--mlf_prior_strength", type=float, default=0.0)
    ap.add_argument("--freq_flatten", action="store_true", default=False)
    ap.add_argument("--flatten_mode", default="avg", choices=["avg", "max"])
    ap.add_argument("--freq_tokenizer", default="k_query", choices=["k_query", "fixed_band"],
                    help="Exp B: k_query (default) | fixed_band (equal-width sub-bands, param-free)")
    ap.add_argument("--preact_fix", dest="preact_fix", action="store_true", default=True)
    ap.add_argument("--no_preact_fix", dest="preact_fix", action="store_false")
    ap.add_argument("--oc_alpha", type=float, default=20.0)
    ap.add_argument("--oc_m_real", type=float, default=0.9)
    ap.add_argument("--oc_m_fake", type=float, default=0.2)
    ap.add_argument("--score_mode", default="raw", choices=["raw", "oc"])

    # runtime
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--eval_len", type=int, default=64600)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--fp32", action="store_true", default=False,
                    help="disable AMP; removes one source of run-to-run variance")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    excluded = set()
    if args.exclude_utts:
        p = Path(args.exclude_utts)
        excluded = {Path(l.split()[0]).stem for l in p.read_text(encoding="utf-8").splitlines()
                    if l.strip() and not l.startswith("#")}
        print(f"[exclude] {len(excluded)} utterance ids from {p}")

    # ---- model ------------------------------------------------------------------
    model = build_model(args, device)
    sd, ckpt_info = load_and_maybe_average(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ckpt][WARN] missing={list(missing)}")
        print(f"[ckpt][WARN] unexpected={list(unexpected)}")
        if missing:
            raise RuntimeError("checkpoint does not match the model. Did the training run use "
                               "different --d_model/--n_layer/--freq_flatten flags?")
    print(f"[ckpt] loaded: {ckpt_info}")

    wanted = [s.strip().lower() for s in args.sets.split(",") if s.strip()]
    summary = {"tag": args.tag, "checkpoints": ckpt_info, "sets": {}}
    lines = []

    for which in wanted:
        if which == "df":
            tr = load_asvspoof_trials(args.data_2021df_meta, args.df_eval_phase, "2021DF-eval")
            audio_dir, ext = args.data_2021df_flac, ".flac"
        elif which == "la":
            tr = load_asvspoof_trials(args.data_2021la_meta, args.la_eval_phase, "2021LA-eval")
            audio_dir, ext = args.data_2021la_flac, ".flac"
        elif which == "itw":
            if not args.itw_dir:
                print("[ITW] skipped: --itw_dir not given")
                continue
            meta = args.itw_meta or str(Path(args.itw_dir) / "meta.csv")
            tr = load_itw_trials(meta)
            audio_dir, ext = args.itw_dir, ".wav"
        else:
            raise ValueError(f"unknown set {which!r}")

        n_dropped = tr.drop(excluded)
        if n_dropped:
            print(f"[{tr.name}] excluded {n_dropped} trial(s) via --exclude_utts")
        if args.limit:
            tr.keys = tr.keys[: args.limit]
            print(f"[{tr.name}][SMOKE] limited to {len(tr)} trials -- numbers are NOT publishable")

        keys, scores, bad = score_partition(model, tr, audio_dir, ext, args, device)
        labels = np.array([tr.label[k] for k in keys], dtype=int)
        eer = eer_of(labels, scores)
        n_bona = int((labels == 1).sum())
        n_spoof = int((labels == 0).sum())
        sep = float(scores[labels == 1].mean() - scores[labels == 0].mean())

        # official-format score file: "<utt> <score>"
        sf = out / f"{args.tag}_{tr.name}.scores.txt"
        with sf.open("w", encoding="utf-8") as f:
            for k, s in zip(keys, scores):
                f.write(f"{k} {s:.8f}\n")

        entry = {"N": len(keys), "bonafide": n_bona, "spoof": n_spoof,
                 "EER_percent": eer, "separation": sep,
                 "undecodable_excluded": len(bad), "score_file": str(sf)}

        # min t-DCF for 2021LA (needs the ASV score/key files)
        if which == "la":
            keys_dir = Path(args.data_2021la_keys_dir) if args.data_2021la_keys_dir \
                else Path(args.data_2021la_meta).parent.parent
            try:
                import evaluate_2021_LA as e2021
                e2021.phase = args.la_eval_phase
                e2021.asv_key_file = str(keys_dir / "ASV" / "trial_metadata.txt")
                e2021.asv_scr_file = str(keys_dir / "ASV" / "ASVTorch_Kaldi" / "score.txt")
                e2021.cm_key_file = str(args.data_2021la_meta)
                if Path(e2021.asv_key_file).exists() and Path(e2021.asv_scr_file).exists():
                    entry["min_tDCF"] = float(e2021.eval_to_score_file(str(sf), str(args.data_2021la_meta)))
                else:
                    print("[2021LA] min t-DCF skipped: ASV key/score files not found under", keys_dir / "ASV")
            except Exception as exc:
                print(f"[2021LA] min t-DCF failed: {exc}")

        # breakdowns (paper table: EER by codec and by attack)
        if which in ("df", "la"):
            for gname, gmap in (("codec", tr.codec), ("attack", tr.attack)):
                rows = breakdown(keys, scores, tr, gmap)
                if not rows:
                    continue
                csv_path = out / f"{args.tag}_{tr.name}_by_{gname}.csv"
                with csv_path.open("w", encoding="utf-8") as f:
                    f.write(f"{gname},n_spoof,EER_percent\n")
                    for g, n, e in rows:
                        f.write(f"{g},{n},{e:.4f}\n")
                entry[f"by_{gname}"] = {g: round(e, 3) for g, n, e in rows}
                print(f"\n[{tr.name}] EER by {gname}:")
                for g, n, e in rows:
                    print(f"    {g:<24s} n={n:>7,d}   EER={e:6.2f}%")

        if bad:
            bf = out / f"{args.tag}_{tr.name}_undecodable.txt"
            bf.write_text("\n".join(bad), encoding="utf-8")
            print(f"[{tr.name}][WARN] {len(bad)} undecodable trial(s) dropped -> {bf}")

        summary["sets"][tr.name] = entry
        lines.append(f"{tr.name:<14s} N={len(keys):>8,d}  bona={n_bona:>7,d}  spoof={n_spoof:>8,d}  "
                     f"EER={eer:6.2f}%" + (f"  min-tDCF={entry['min_tDCF']:.4f}" if "min_tDCF" in entry else ""))

    print("\n" + "=" * 78)
    print(f"FULL-PARTITION RESULT   tag={args.tag}   checkpoints={ckpt_info['n_ckpt']}")
    print("=" * 78)
    for l in lines:
        print("  " + l)
    print("=" * 78)
    if args.limit:
        print("!! --limit was set: these are SMOKE-TEST numbers, not publishable.")

    (out / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / f"{args.tag}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary -> {out / (args.tag + '_summary.json')}")


if __name__ == "__main__":
    main()
