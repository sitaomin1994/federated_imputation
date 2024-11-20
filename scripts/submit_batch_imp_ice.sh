#!/bin/bash
configs=(
  "hhpct_exp11"
  "hhpct_exp12"
  "hhpct_exp13"
  "hhpct_exp21"
  "hhpct_exp22"
  "hhpct_exp23"
#  "vehicle_exp11"
#  "vehicle_exp12"
#  "vehicle_exp13"
#  "vehicle_exp21"
#  "vehicle_exp22"
#  "vehicle_exp23"
#  "eicumo_exp11"
#  "eicumo_exp12"
#  "eicumo_exp13"
#  "eicumo_exp21"
#  "eicumo_exp22"
#  "eicumo_exp23"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_imp.sh "${config_name}"
done