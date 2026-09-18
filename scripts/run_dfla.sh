#!/usr/bin/env bash
# 2021 DF + LA route. RawBoost 1,2,3; checkpoint selection on official 2019LA dev;
# final test on the FULL official 2021DF and 2021LA eval partitions.
set -e
DATA=${1:?usage: run_dfla.sh DATA_ROOT OUT_DIR}
OUT=${2:-./exp_dfla}
python3 scripts/train.py --route dfla -o "$OUT" \
  --data_2019la "$DATA/ASVspoof2019" \
  --data_2021df_flac "$DATA/2021DF/ASVspoof2021_DF_eval/flac" \
  --trial_metadata "$DATA/2021DF/keys/CM/trial_metadata.txt" \
  --data_2021la_flac "$DATA/2021LA/ASVspoof2021_LA_eval/flac" \
  --data_2021la_meta "$DATA/2021LA/keys/CM/trial_metadata.txt" \
  --rawboost_algos 1,2,3 \
  --epochs 30 --batch_size 64 --eval_batch_size 256 \
  --base_lr 5e-4 --warmup 3000 --grad_clip 0.5 --weight_decay 1e-4 \
  --rank_lambda 0 --rank_margin 0 --oc_lambda 0.02 \
  --aux_start_epoch 0 --aux_warmup_epochs 0 \
  --num_workers 8
