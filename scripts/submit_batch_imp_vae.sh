#!/bin/bash
configs=(
  "hhpct_exp14"
  "hhpct_exp24"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp_vae.sh "${config_name}"
done