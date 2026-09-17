#!/usr/bin/env python3
"""Offline legacy min-tDCF for ASVspoof2019 LA from a 2-column score file.

Replicates scripts/eval_any_ckpt.py exactly (compute_tDCF_legacy + COST_MODEL),
but reads an existing <utt> <score> file so no GPU/model run is needed.

Usage:
  python scripts/legacy_tdcf_from_scores.py \
    --scores experiments/phase0_recheck/19la/2019LA_eval_scores.best_19la.txt \
    --tag best_19la
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eval_metric_LA as em

DATA = Path("/root/autodl-tmp/data/2019LA")
PROTO = DATA / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
ASV_EVAL = DATA / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
ASV_DEV = DATA / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"

COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": 0.95 * 0.99,
    "Pnon": 0.95 * 0.01,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}


def load_scores(path):
    sc = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) == 2:
            sc[p[0]] = float(p[1])
    return sc


def load_protocol():
    lab = {}
    for line in PROTO.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        lab[p[1]] = 1 if p[4].lower() == "bonafide" else 0
    return lab


def load_asv(path):
    tar, non, spf = [], [], []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        lab = p[1].lower()
        try:
            v = float(p[-1])
        except ValueError:
            continue
        if lab == "target":
            tar.append(v)
        elif lab == "nontarget":
            non.append(v)
        elif lab == "spoof":
            spf.append(v)
    return np.asarray(tar), np.asarray(non), np.asarray(spf)


def legacy_tdcf(bona, spoof, pfa, pmiss, pmiss_spoof):
    curve, _ = em.compute_tDCF_legacy(
        np.asarray(bona), np.asarray(spoof), pfa, pmiss, pmiss_spoof, COST_MODEL, False)
    return float(np.min(curve))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sc = load_scores(args.scores)
    lab = load_protocol()
    keys = [k for k in sc if k in lab]
    y = np.asarray([lab[k] for k in keys])
    s = np.asarray([sc[k] for k in keys])
    bona, spoof = s[y == 1], s[y == 0]
    eer, _ = em.compute_eer(bona, spoof)

    tar_e, non_e, spf_e = load_asv(ASV_EVAL)
    _, thr_e = em.compute_eer(tar_e, non_e)
    pfa, pmiss, pmiss_spoof, _ = em.obtain_asv_error_rates(tar_e, non_e, spf_e, thr_e)
    tdcf_eval = legacy_tdcf(bona, spoof, pfa, pmiss, pmiss_spoof)

    tar_d, non_d, _ = load_asv(ASV_DEV)
    _, thr_d = em.compute_eer(tar_d, non_d)
    pfa_d, pmiss_d, pmiss_spoof_d, _ = em.obtain_asv_error_rates(tar_e, non_e, spf_e, thr_d)
    tdcf_dev = legacy_tdcf(bona, spoof, pfa_d, pmiss_d, pmiss_spoof_d)

    out = {
        "tag": args.tag,
        "scores": args.scores,
        "n_scored": int(len(keys)),
        "n_bonafide": int(bona.size),
        "n_spoof": int(spoof.size),
        "eer_percent": float(eer) * 100.0,
        "min_tdcf_legacy_eval_asv": tdcf_eval,
        "min_tdcf_legacy_dev_asv": tdcf_dev,
        "cost_model": COST_MODEL,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("->", args.out)


if __name__ == "__main__":
    main()
