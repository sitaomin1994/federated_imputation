# Complementarity Adjusted Federated (Cafe) Imputation

This is the code repo for paper - Cafe: Improved Federated Data Imputation by Leveraging Missing Data Heterogeneity

## Setup Environment and Data


Run following command to fetch `Codon` and `Codrna` dataset, this will download data to data folder.

## Federated Imputation

Run following command to simulate heterogenenous missing data FL scenarios and perform federated imputation with Cafe and baselines

After finishing running, the results will be stored in json files under a folder of raw_results folder. You can open `./notebook/results_analysis.ipynb` and run code to display the results in a more readable format.

## Federated Prediction

Run following command to perform federated prediction of imputed datasets.
