#!/bin/bash
configs=(
#  "vehicle_exp11"
#  "vehicle_exp12"
#  "vehicle_exp13"
#  "vehicle_exp21"
#  "vehicle_exp22"
  "vehicle_exp23"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job_name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp.sh "${config_name}"
done