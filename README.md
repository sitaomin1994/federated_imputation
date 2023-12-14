# Complementarity Adjusted Federated (Cafe) Imputation

This is the code repo for paper - Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity

## Setup Environment and Data

Setup conda environment in terminal
```
conda env create -f environment.yml
conda activate fed_imp
```

Set up project structure

```
python setup_project.py
```

Download dataset to data folder
```
python scripts/download_data.py
```


## Federated Imputation

Run following command to simulate heterogenenous missing data FL scenarios and perform federated imputation with Cafe and baselines.

```
python run_fed_imputation.py
```

This script will run following the configuration in `conf/exp_config_imp.yaml` file, it loads the `conf/cnofig_vars/exp_demo.yaml` which includes configuration of data partition strategy, missing data simulation strategy and imputation methods, etc.  

By default, it uses multiprocessing to speed up, you can set `mtp` to `false` in config files to disable it. By default, it will run all scenarios `ideal`, `random`, `s1-s4` for 3 baselines and cafe, you can change it in config to run specific scenarios and baselines. 

After finishing running, the results will be stored in json files under a folder of `\results\raw_results` folder. Run the following script to process resuls into a excel file stored in `\results\processed_results`. 

```
python scripts/results_fedimp.py
```

We give a `./notebook/demo.ipynb` to show the analysis of imputation results in a more readable format.

## Federated Prediction

Run following command to perform federated prediction of imputed datasets of each clients.

```
pythion run_fed_prediction.py
```

After finishing running, the results will be stored in json files under a folder of `\results\raw_results` folder. Run the following script to process resuls into a excel file stored in `\results\processed_results`. 

```
python scripts/results_fedpred.py
```

We give a `./notebook/demo.ipynb` to show the analysis of prediction results in a more readable format.


