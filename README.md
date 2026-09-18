# FALG-Mamba — experiment code

Lightweight multi-level frequency-aware state-space model for raw-waveform audio
deepfake detection. This release contains only the **best configuration**; the
backend / ablation / layer-selection variants used for the analysis are not
included.

## Released configuration

| Component | Setting |
|---|---|
| Frequency tokenizer | learnable K-query (`FreqAwareCompressionTK`), K = 4 |
| Cross-level fusion | token-wise attention (`attn`), time grid = `middle`, prior strength 0 |
| Backbone | 6 × (depthwise conv + Mamba-2) with **alternating scan** (`alt`) |
| Front end | multi-level Sinc + SE-Res2Net, pre-activation fix |
| Loss | weighted CE (0.1/0.9) + OC-Softmax (λ = 0.02); ranking disabled |
| Score | raw classifier logits, `score = logits[:,1] - logits[:,0]` |
| Params | 0.50 M |

Auxiliary-loss warmup is disabled in the default trainer and all released run
scripts. OC-Softmax uses its fixed coefficient of 0.02 from epoch 1, and the
pairwise ranking coefficient is 0. Ranking is skipped when its coefficient is
zero. This is separate from the learning-rate warmup controlled by `--warmup`.

## Protocol

* **Training data** — full ASVspoof 2019 LA *train* (25,380), no class-ratio
  sampling.
* **Checkpoint selection** — the **full official ASVspoof 2019 LA dev**
  partition; the checkpoint with the **lowest dev EER** is kept
  (`best_model.pt`).
* **Testing** — full official partitions only, no subsampling:
  * `--route la19` → ASVspoof 2019 LA **eval**
  * `--route dfla` → ASVspoof 2021 **DF eval** + ASVspoof 2021 **LA eval**
* **RawBoost** — algorithm list `1,2,3` for the `dfla`/`itw` routes; **off**
  for the `la19` route (`--rawboost_algos ""`).

> The earlier training variants selected checkpoints on held-out subsets of the
> *evaluation* sets (2021DF/2021LA 20%/30%) or on the test partition itself.
> This release removes that and uses dev only.

## Layout

```
FALG-Mamba_code/
  model_scripts/AA_LG_Mamba_MLF_PG.py   # model (best config only)
  myBlocks.py  resnet_blocks.py  oc_softmax.py
  RawBoost.py
  data_utils.py  data_utils_2021df.py  data_utils_eval_extra.py
  eval_metric_LA.py  eval_metrics_DF.py
  scripts/
    train.py                  # unified trainer (-e) for la19 / dfla
    train_itw.py              # ITW-route trainer
    eval_full.py              # evaluate a checkpoint on full DF/LA/ITW
    eval_2019la.py  eval_itw.py  itw_validation.py
    legacy_tdcf_from_scores.py
    run_dfla.sh  run_la19.sh  run_itw.sh
```

## Install

```bash
pip install -r requirements.txt
# Mamba-2 kernels (match your CUDA/torch):
pip install causal-conv1d>=1.4.0 mamba-ssm>=2.2.0
```

## Data layout

```
DATA/
  ASVspoof2019/
    ASVspoof2019_LA_train/flac/
    ASVspoof2019_LA_dev/flac/
    ASVspoof2019_LA_eval/flac/
    ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train,dev,eval}.trl.txt
  2021DF/
    ASVspoof2021_DF_eval/flac/
    keys/CM/trial_metadata.txt
  2021LA/
    ASVspoof2021_LA_eval/flac/
    keys/CM/trial_metadata.txt
  release_in_the_wild/            # ITW: *.wav + meta.csv
```

## Run

```bash
# 2021 DF + LA route (RawBoost 1,2,3; dev selection; full DF/LA test)
bash scripts/run_dfla.sh /path/to/DATA ./exp_dfla

# 2019 LA route (no RawBoost; dev selection; full 2019LA eval test)
bash scripts/run_la19.sh /path/to/DATA ./exp_la19

# In-the-Wild route
bash scripts/run_itw.sh /path/to/DATA ./exp_itw
```

Outputs: `train_log.csv` (per-epoch dev EER), `best_model.pt`,
`2021DF_eval_scores.best.txt` / `2021LA_eval_scores.best.txt` (or
`2019LA_eval_scores.best.txt`), `final_result.txt`.

## Notes

* `eval_full.py` (in `scripts/`) evaluates any checkpoint on the full official
  DF / LA / ITW partitions and writes official-format score files.
* 2019 LA  min-tDCF can be recomputed offline from a score file with
  `scripts/tdcf_from_scores.py`.
* The model score is an unbounded log-likelihood-ratio-like value; no score
  normalization is applied.

Code will be released upon acceptance.
