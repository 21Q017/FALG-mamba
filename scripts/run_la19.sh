#!/usr/bin/env bash
# 2019 LA route. RawBoost OFF; selection on official 2019LA dev; final test on the
# FULL official 2019LA eval partition.
set -e
DATA=${1:?usage: run_la19.sh DATA_ROOT OUT_DIR}
OUT=${2:-./exp_la19}
python3 scripts/train.py --route la19 -o "$OUT" \
  --data_2019la "$DATA/ASVspoof2019" \
  --rawboost_algos "" \
  --epochs 60 --batch_size 64 --eval_batch_size 128 \
  --base_lr 5e-4 --warmup 0 --grad_clip 0.5 --weight_decay 1e-4 \
  --rank_lambda 0 --rank_margin 0 --oc_lambda 0.02 \
  --aux_start_epoch 0 --aux_warmup_epochs 0 \
  --num_workers 8
