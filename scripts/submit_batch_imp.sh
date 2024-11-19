#!/bin/bash
configs=(
#  "eicumo_exp11"
#  "eicumo_exp12"
#  "eicumo_exp13"
  "eicumo_exp21"
  "eicumo_exp22"
  "eicumo_exp23"
#  "heart_disease_exp11"
#  "heart_disease_exp12"
#  "heart_disease_exp13"
#  "heart_disease_exp21"
#  "heart_disease_exp22"
#  "heart_disease_exp23"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp.sh "${config_name}"
done