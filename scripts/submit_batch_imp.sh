#!/bin/bash
configs=(
  "hhnpct_exp11"
  "hhnpct_exp12"
  "hhnpct_exp13"
  "hhnpct_exp21"
  "hhnpct_exp22"
  #"hhnpct_exp23"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp.sh "${config_name}"
done