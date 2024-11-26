#!/bin/bash
configs=(
#  "1120/hhp_ct3"
#  "1120/vehicle2"
#  "1120/eicu_mo2"
#  "1120/hhp_ct2"
#  "1120/eicu_mo1"
  "1125/eicu_mo2"
  "1125/hhp_ct2"
)

for config_name in "${configs[@]}"; do
    echo "Submitting job for ${config_name}"
    sbatch --job-name="${config_name}" ~/Research/federated_imputation/scripts/submit_job_pred.sh "${config_name}"
done