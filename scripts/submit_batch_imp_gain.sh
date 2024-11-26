#!/bin/bash
configs=(
  "hhpct_exp14"
  "hhpct_exp24"
  "eicumo_exp14"
  "eicumo_exp24"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp_gain.sh "${config_name}"
done
