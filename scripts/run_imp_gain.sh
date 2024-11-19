#!/bin/bash

config_name=$1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# CAFE Natural Partition
# e.g. python main_experiment_amarel.py config_vars/imp_extention2=hhplos_exp1.yaml
echo "Executing command: python main_experiment_gain.py config_vars/imp_extention2=${config_name}.yaml"
python main_experiment_gain.py config_vars/imp_extention2="${config_name}".yaml


# CAFE Test
#python main_experiment_amarel.py config_vars/imp_extention2=hhplos_exp_test.yaml