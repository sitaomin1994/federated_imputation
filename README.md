# Complementarity Adjusted Federated (Cafe) Imputation

This is the code repo for paper - Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity

## Setup Environment and Data


Run following command to fetch `Codon` dataset, this will download data to data folder.

## Federated Imputation

Run following command to simulate heterogenenous missing data FL scenarios and perform federated imputation with Cafe and baselines.

```
python run_fed_imputation.py
```

This script will run following the configuration in `conf/exp_config_imp.yaml` file, it loads the `conf/cnofig_vars/exp_demo.yaml` which includes configuration of data partition strategy, missing data simulation strategy and imputation methods, etc. 

After finishing running, the results will be stored in json files under a folder of `\results\raw_results` folder with date as folder name. Run the following script to process resuls into a excel file stored in `\results\processed_results`. 

We also give a `./notebook/demo.ipynb` to display missing data scenarios and the analysis of imputation results in a more readable format.

## Federated Prediction

Run following command to perform federated prediction of imputed datasets of each clients.

```
pythion run_fed_prediction.py
```




