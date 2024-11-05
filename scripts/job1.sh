#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# CAFE Natural Partition
#python main_experiment_amarel.py config_vars/imp_extention2=hhplos_exp1.yaml

# CAFE Test
python main_experiment_amarel.py config_vars/imp_extention2=hhplos_exp_test.yaml