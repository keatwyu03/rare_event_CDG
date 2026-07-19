#!/bin/bash
# ============================================================================
# Launch the rate-shock conditional diffusion pipeline.
#
#   bash run.sh                 # full pipeline (pretrain -> hfunction -> sample)
#   bash run.sh stats           # print per-stock mean/std (train/test, all/event); no training
#   bash run.sh pretrain        # train the diffusion backbone only
#   bash run.sh hfunction       # train the h-function only (independent of backbone)
#   bash run.sh sample          # load both ckpts, conditional generation + plots
#
#   GPU=1 bash run.sh           # pick CUDA device 1
#
# All key training knobs live here — edit and relaunch.
# ============================================================================
set -e
cd "$(dirname "$0")"

STAGE="${1:-all}"

# ---- GPU selection (cdg_finance-style) ----
# CUDA_VISIBLE_DEVICES masks to one physical GPU, which the process then sees as
# cuda:0. This is the ONLY selection mechanism (do NOT also pass --gpu, or the
# code would ask for cuda:$GPU which no longer exists -> "Invalid device id").
GPU="${GPU:-0}"                       # physical CUDA device index to use
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

# Every param below uses ${VAR:-default}, so ANY of them can be overridden from
# the environment, e.g.:  GPU=2 H_T_MAX=0.8 GAMMA=0.5 bash run.sh hfunction
# (a plain `VAR=...` assignment would IGNORE the env var — don't use that here).

# ---- data ----
CSV_PATH="${CSV_PATH:-$HOME/Desktop/tech_stocks_tips.csv}"
SEED="${SEED:-0}"
START_DATE=""   # data window (YYYY-MM-DD); set empty for all data
END_DATE=""       # pick a stable window with: python select_window.py
EVENT_QUANTILE="${EVENT_QUANTILE:-0.90}" # Δy quantile = event; 0.90=top10%, 0.99=rarer.
                                         # h ckpt/figures are keyed by this (q90, q99, ...).

# ---- pretrain: score backbone (richer net — score matching is the harder task) ----
PRE_D_MODEL="${PRE_D_MODEL:-256}"
PRE_N_HEADS="${PRE_N_HEADS:-16}"
PRE_N_LAYERS="${PRE_N_LAYERS:-8}"
PRE_DIM_FF="${PRE_DIM_FF:-512}"
PRE_DROPOUT="${PRE_DROPOUT:-0.0}"
PRE_EPOCHS="${PRE_EPOCHS:-500}"
PRE_BATCH="${PRE_BATCH:-256}"
PRE_LR="${PRE_LR:-1e-4}"

# ---- h-function: time-dependent classifier (lighter net) ----
H_D_MODEL="${H_D_MODEL:-256}"
H_N_HEADS="${H_N_HEADS:-8}"
H_N_LAYERS="${H_N_LAYERS:-6}"
H_DIM_FF="${H_DIM_FF:-256}"
H_DROPOUT="${H_DROPOUT:-0.0}"
H_EPOCHS="${H_EPOCHS:-500}"
H_BATCH="${H_BATCH:-256}"
H_LR="${H_LR:-1e-4}"
H_T_MAX="${H_T_MAX:-1}"        # only t in [eps0, H_T_MAX] (near clean data) used to
                                 # train h / apply guidance; ckpt is keyed by this value.
H_POS_WEIGHT="${H_POS_WEIGHT:--1}"   # BCE positive-class weight for the ~10% imbalance.
                                 # <=0 = auto (#neg/#pos ≈ 9); positive value overrides.

# ---- sampling ----
N_SAMPLE="${N_SAMPLE:-10000}"    # M samples for histograms / correlation
N_STEPS="${N_STEPS:-100}"        # reverse Euler-Maruyama steps
SAMPLE_BATCH="${SAMPLE_BATCH:-1000}"   # sampling mini-batch (lower if GPU OOM)
GAMMA="${GAMMA:-2.0}"            # Doob guidance strength (1.0 = exact)

python -u main.py \
    --stage     "$STAGE" \
    --csv-path  "$CSV_PATH" \
    --start-date "$START_DATE" \
    --end-date   "$END_DATE" \
    --event-quantile "$EVENT_QUANTILE" \
    --seed      "$SEED" \
    --pre-d-model  "$PRE_D_MODEL" \
    --pre-n-heads  "$PRE_N_HEADS" \
    --pre-n-layers "$PRE_N_LAYERS" \
    --pre-dim-ff   "$PRE_DIM_FF" \
    --pre-dropout  "$PRE_DROPOUT" \
    --pre-epochs "$PRE_EPOCHS" \
    --pre-batch-size "$PRE_BATCH" \
    --pre-lr    "$PRE_LR" \
    --h-d-model  "$H_D_MODEL" \
    --h-n-heads  "$H_N_HEADS" \
    --h-n-layers "$H_N_LAYERS" \
    --h-dim-ff   "$H_DIM_FF" \
    --h-dropout  "$H_DROPOUT" \
    --h-epochs  "$H_EPOCHS" \
    --h-batch-size "$H_BATCH" \
    --h-lr      "$H_LR" \
    --h-t-max   "$H_T_MAX" \
    --h-pos-weight "$H_POS_WEIGHT" \
    --n-sample  "$N_SAMPLE" \
    --n-steps   "$N_STEPS" \
    --sample-batch "$SAMPLE_BATCH" \
    --gamma     "$GAMMA"
