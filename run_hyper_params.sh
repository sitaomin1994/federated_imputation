#!/bin/bash

# Define the list of datasets
datasets=("heart", "mimiciii_mo2", "mimiciii_icd") # Replace with your actual dataset names

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Loop through each mechanism
    # Run the Python script with current dataset and mechanism
    python hyper_params_tune.py --dataset "$dataset" --s s2 --mm even
    done
done

#datasets=("mimiciii_icd") # Replace with your actual dataset names
#
## Loop through each dataset
#for dataset in "${datasets[@]}"; do
#    # Loop through each mechanism
#    for mech in "lr"; do
#        # Run the Python script with current dataset and mechanism
#        python hyper_params_tune.py --dataset "$dataset" --s s1 --mm "$mech"
#    done
#done
#
#datasets=("codrna") # Replace with your actual dataset names
#
## Loop through each dataset
#for dataset in "${datasets[@]}"; do
#    # Loop through each mechanism
#    for mech in "lr" "rl"; do
#        # Run the Python script with current dataset and mechanism
#        python hyper_params_tune.py --dataset "$dataset" --s s1 --mm "$mech"
#    done
#done