#!/usr/bin/env bash
# In-the-Wild route. RawBoost 1,2,3; final test on the full ITW set.
set -e
DATA=${1:?usage: run_itw.sh DATA_ROOT OUT_DIR}
OUT=${2:-./exp_itw}
python3 scripts/train_itw.py --itw_only_val -o "$OUT" \
  --data_2019la "$DATA/ASVspoof2019" \
  --itw_root "$DATA/release_in_the_wild" \
  --rawboost_algos 1,2,3 \
  --epochs 60 --batch_size 64 --eval_batch_size 256 \
  --base_lr 1e-4 --warmup 300 --grad_clip 0.5 --weight_decay 1e-4 \
  --rank_lambda 0 --rank_margin 0 --oc_lambda 0.02 \
  --aux_start_epoch 0 --aux_warmup_epochs 0 \
  --num_workers 8
