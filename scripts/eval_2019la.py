#!/usr/bin/env python3
"""Full ASVspoof 2019 LA dev/eval evaluation for FALG-Mamba variants.

The script follows the score convention used by ``train.py``:
spoof=0, bonafide=1, and a larger ``logits[:, 1] - logits[:, 0]`` score means
more bonafide. It supports one checkpoint or fixed checkpoint averaging.

EER is always reported. min t-DCF is reported when the official ASV score file
can be found below ``--data_2019la`` or is supplied with ``--asv_score_file``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import eval_metric_LA as em
from data_utils_eval_extra import BAD_AUDIO_SENTINEL, Dataset_Audio_eval
from train import autocast_context, bonafide_score_from_logits, build_model


COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": 0.95 * 0.99,
    "Pnon": 0.95 * 0.01,
    "Cmiss": 1,
    "Cfa": 10,
    "Cfa_spoof": 10,
}


def load_protocol(path: Path) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"2019 LA protocol not found: {path}")
    keys: List[str] = []
    labels: Dict[str, int] = {}
    attacks: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            label = next((x.lower() for x in reversed(parts) if x.lower() in {"bonafide", "spoof"}), None)
            if label is None:
                continue
            utt = Path(parts[1]).stem
            attack = "bonafide" if label == "bonafide" else (parts[-2] if len(parts) >= 2 else "-")
            keys.append(utt)
            labels[utt] = 1 if label == "bonafide" else 0
            attacks[utt] = attack
    if not keys:
        raise RuntimeError(f"No labelled trials found in {path}")
    return keys, labels, attacks


def protocol_path(root: Path, partition: str) -> Path:
    return root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{partition}.trl.txt"


def audio_dir(root: Path, partition: str) -> Path:
    base = root / f"ASVspoof2019_LA_{partition}"
    return base / "flac" if (base / "flac").is_dir() else base


def load_and_average(paths: Iterable[str]) -> Tuple[dict, dict]:
    paths = [str(Path(p)) for p in paths]
    states = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            payload = payload["model"]
        elif isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if payload and all(str(k).startswith("module.") for k in payload):
            payload = {str(k)[7:]: v for k, v in payload.items()}
        states.append(payload)
    ref = states[0]
    for i, state in enumerate(states[1:], 1):
        if set(state) != set(ref):
            raise RuntimeError(f"checkpoint architecture mismatch: {paths[0]} vs {paths[i]}")
    if len(states) == 1:
        return ref, {"n_ckpt": 1, "checkpoints": paths}
    avg = {}
    for key, value in ref.items():
        if torch.is_floating_point(value):
            avg[key] = torch.stack([state[key].float() for state in states]).mean(0).to(value.dtype)
        else:
            avg[key] = value.clone()
    return avg, {"n_ckpt": len(states), "checkpoints": paths}


def score_model(model, loader, device, args):
    model.eval()
    keys_all: List[str] = []
    scores_all: List[float] = []
    unreadable: List[str] = []
    with torch.no_grad():
        for data, keys in tqdm(loader, desc="2019LA score", ncols=100):
            key_list = [str(k) for k in keys]
            keep = [i for i, key in enumerate(key_list) if not key.startswith(BAD_AUDIO_SENTINEL)]
            unreadable.extend(key[len(BAD_AUDIO_SENTINEL):] for key in key_list if key.startswith(BAD_AUDIO_SENTINEL))
            if not keep:
                continue
            data = data.to(device, non_blocking=True)
            with autocast_context(args, device):
                out = model(data)
                logits = out[0] if isinstance(out, tuple) else out
            if len(keep) != len(key_list):
                index = torch.tensor(keep, device=logits.device)
                logits = logits.index_select(0, index)
                key_list = [key_list[i] for i in keep]
            scores = bonafide_score_from_logits(logits).detach().float().cpu().numpy()
            keys_all.extend(key_list)
            scores_all.extend(float(x) for x in scores)
    return keys_all, np.asarray(scores_all, dtype=float), unreadable


def find_asv_score_file(root: Path, partition: str) -> Optional[Path]:
    names = [
        f"ASVspoof2019.LA.asv.{partition}.gi.trl.scores.txt",
        f"ASVspoof2019.LA.asv.{partition}.trl.scores.txt",
    ]
    for name in names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]
    return None


def load_asv_scores(path: Path):
    target, nontarget, spoof = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            label = next((x.lower() for x in parts if x.lower() in {"target", "nontarget", "spoof"}), None)
            if label is None:
                continue
            score = None
            for token in reversed(parts):
                try:
                    score = float(token)
                    break
                except ValueError:
                    continue
            if score is None:
                continue
            {"target": target, "nontarget": nontarget, "spoof": spoof}[label].append(score)
    arrays = tuple(np.asarray(x, dtype=float) for x in (target, nontarget, spoof))
    if any(x.size == 0 for x in arrays):
        raise RuntimeError(
            f"Could not parse target/nontarget/spoof ASV scores from {path}; "
            f"counts={[int(x.size) for x in arrays]}"
        )
    return arrays


def compute_metrics(keys, scores, labels, attacks, asv_score_file: Optional[Path]):
    label_arr = np.asarray([labels[k] for k in keys], dtype=int)
    bona = scores[label_arr == 1]
    spoof = scores[label_arr == 0]
    eer, threshold = em.compute_eer(bona, spoof)
    result = {
        "n_trials": int(len(keys)),
        "n_bonafide": int(len(bona)),
        "n_spoof": int(len(spoof)),
        "eer_percent": float(eer * 100.0),
        "eer_threshold": float(threshold),
        "min_tDCF": None,
        "per_attack_eer_percent": {},
    }
    if asv_score_file is not None and asv_score_file.is_file():
        tar_asv, non_asv, spoof_asv = load_asv_scores(asv_score_file)
        _, asv_threshold = em.compute_eer(tar_asv, non_asv)
        pfa, pmiss, _, pfa_spoof = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)
        curve, _ = em.compute_tDCF(bona, spoof, pfa, pmiss, pfa_spoof, COST_MODEL, False)
        result["min_tDCF"] = float(np.min(curve))
        result["asv_score_file"] = str(asv_score_file)

    for attack in sorted(set(attacks.values())):
        if attack == "bonafide":
            continue
        attack_scores = np.asarray([s for k, s in zip(keys, scores) if labels[k] == 0 and attacks[k] == attack])
        if attack_scores.size == 0:
            continue
        attack_eer, _ = em.compute_eer(bona, attack_scores)
        result["per_attack_eer_percent"][attack] = float(attack_eer * 100.0)
    return result


def add_model_args(ap: argparse.ArgumentParser):
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_query", type=int, default=4)
    ap.add_argument("--scan", choices=["alt", "uni", "bi"], default="alt")
    ap.add_argument("--backend_mode", choices=["gated", "fixed", "mamba", "local"], default="gated")
    ap.add_argument("--sinc_channels", type=int, default=70)
    ap.add_argument("--pool_heads", type=int, default=4)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freq_pool", type=int, default=3)
    ap.add_argument("--levels", default="shallow,middle,deep")
    ap.add_argument("--mlf_align_level", choices=["shallow", "middle", "deep"], default="middle")
    ap.add_argument("--mlf_align_mode", choices=["antialias", "linear", "nearest"], default="antialias")
    ap.add_argument("--mlf_fusion", choices=["attn", "prior_attn", "prior_only", "mean"], default="attn")
    ap.add_argument("--mlf_prior_init", default="0.33,0.33,0.34")
    ap.add_argument("--mlf_prior_strength", type=float, default=0.0)
    ap.add_argument("--freq_flatten", action="store_true", default=False)
    ap.add_argument("--flatten_mode", choices=["avg", "max"], default="avg")
    ap.add_argument("--freq_tokenizer", choices=["k_query", "fixed_band"], default="k_query")
    ap.add_argument("--preact_fix", dest="preact_fix", action="store_true", default=True)
    ap.add_argument("--no_preact_fix", dest="preact_fix", action="store_false")
    ap.add_argument("--score_mode", choices=["raw", "oc"], default="raw")
    ap.add_argument("--oc_alpha", type=float, default=20.0)
    ap.add_argument("--oc_m_real", type=float, default=0.9)
    ap.add_argument("--oc_m_fake", type=float, default=0.2)


def main():
    ap = argparse.ArgumentParser(description="Evaluate FALG-Mamba on full ASVspoof 2019 LA")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--data_2019la", default="/root/autodl-tmp/data/2019LA")
    ap.add_argument("--partition", choices=["dev", "eval", "both"], default="eval")
    ap.add_argument("--out_dir", default="./eval_2019la")
    ap.add_argument("--asv_score_file", default="", help="optional official ASV score file; auto-discovered otherwise")
    ap.add_argument("--eval_len", type=int, default=64600)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=True)
    ap.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    ap.add_argument("--audio_error", choices=["skip", "raise"], default="skip")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="fp16")
    add_model_args(ap)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_fold = str(out_dir)
    args.batch_size = args.eval_batch_size
    args.tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, device)
    state, ckpt_info = load_and_average(args.ckpt)
    model.load_state_dict(state, strict=True)

    root = Path(args.data_2019la)
    partitions = ["dev", "eval"] if args.partition == "both" else [args.partition]
    summary = {"tag": args.tag, "checkpoint": ckpt_info, "model": {
        "levels": args.levels, "freq_flatten": args.freq_flatten,
        "freq_tokenizer": args.freq_tokenizer,
        "mlf_fusion": args.mlf_fusion, "mlf_align_mode": args.mlf_align_mode,
        "mlf_align_level": args.mlf_align_level, "backend_mode": args.backend_mode,
    }, "partitions": {}}

    for partition in partitions:
        keys, labels, attacks = load_protocol(protocol_path(root, partition))
        dataset = Dataset_Audio_eval(
            keys, audio_dir(root, partition), ext=".flac", target_len=args.eval_len,
            name=f"2019LA-{partition}", on_error=args.audio_error,
        )
        kwargs = dict(
            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(), drop_last=False,
        )
        if args.num_workers > 0:
            kwargs["persistent_workers"] = args.persistent_workers
            kwargs["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(dataset, **kwargs)
        scored_keys, scores, unreadable = score_model(model, loader, device, args)

        score_path = out_dir / f"2019LA_{partition}_scores.{args.tag}.txt"
        with score_path.open("w", encoding="utf-8") as f:
            for key, score in zip(scored_keys, scores):
                f.write(f"{key} {score:.8f}\n")

        asv_path = Path(args.asv_score_file) if args.asv_score_file else find_asv_score_file(root, partition)
        metrics = compute_metrics(scored_keys, scores, labels, attacks, asv_path)
        metrics["score_file"] = str(score_path)
        metrics["unreadable"] = unreadable
        summary["partitions"][partition] = metrics
        tdcf = " unavailable" if metrics["min_tDCF"] is None else f"={metrics['min_tDCF']:.6f}"
        print(
            f"[2019LA-{partition}] N={metrics['n_trials']} "
            f"EER={metrics['eer_percent']:.4f}% min-tDCF{tdcf} score={score_path}"
        )

    result_path = out_dir / f"2019LA_result.{args.tag}.json"
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] result={result_path}")


if __name__ == "__main__":
    main()
