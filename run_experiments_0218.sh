#!/bin/bash
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#python main_experiment_vae.py
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#python main_experiment_gain.py
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python run_pred_experiments4.py
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#python run_pred_experiment41.py