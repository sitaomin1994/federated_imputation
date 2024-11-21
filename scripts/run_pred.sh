#!/bin/bash
dataset=$1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# prediction
python run_pred_experiments_ext2.py \
    --dir_name fed_imp_ext2_amarel \
    --datasets "${dataset}" \
    --tasks clf \
    --mechanism random2@mrl=0.3_mrr=0.7_mm=mnarlrq,random2@mrl=0.3_mrr=0.7_mm=mnarlrsigst \
    --methods fedavg-s,fedmechw_new,local \
    --n_rounds 10 --n_jobs -1 --mtp True
