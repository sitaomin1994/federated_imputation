#!/bin/bash

# Define the list of datasets
datasets=("genetic") # Replace with your actual dataset names

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Loop through each mechanism
    for mech in "lr" "rl"; do
        # Run the Python script with current dataset and mechanism
        python hyper_params_tune.py --dataset "$dataset" --s s1 --mm "$mech"
    done
done