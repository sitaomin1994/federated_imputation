#!/bin/bash
configs=(
  "hhpct_exp11"
  "hhpct_exp12"
  "hhpct_exp13"
  "hhpct_exp21"
  "hhpct_exp22"
  #"hhnpct_exp23"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp.sh "${config_name}"
done