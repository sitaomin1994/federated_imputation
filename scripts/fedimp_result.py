import os
import json
import pandas as pd
import sys

def process_ret(data, scenario, mr, n_clients, sample_size):
    dir_path = "./results/raw_results/{}/{}/sample@p={}/{}/{}/".format(data, n_clients, sample_size, scenario, mr)
    # Read all json files from directory
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.json')]
    # Read all json files
    df_data = []
    for f in files:
        with open(dir_path+f, 'r') as fp:
            ret = {}
            data = json.load(fp)
            ret['scenario'] = scenario
            ret['mr'] = mr
            ret['method'] = data["params"]["config"]["agg_strategy_imp"]["strategy"]
            ret['rmse'] = data["results"]["avg_imp_final"]["imp@rmse"]
            ret['ws'] = data["results"]["avg_imp_final"]["imp@w2"]
            ret['sliced_ws'] = data["results"]["avg_imp_final"]["imp@sliced_ws"]
            # ret['fed_accu'] = data["results"]["avg_pred_final_model1"]["accu"]
            # ret['fed_f1'] = data["results"]["avg_pred_final_model1"]["f1"]
            # ret['fed_accu_std'] = data["results"]["avg_pred_final_model1"]["accu_std"]
            # ret['fed_f1_std'] = data["results"]["avg_pred_final_model1"]["f1_std"]
            df_data.append(ret)
    
    df = pd.DataFrame(df_data)
    sort_order = ['local', 'fedavg', 'fedavg-s', 'fedwavg', 'fedmechw', 'fedmechclw', 'fedwavgcl', 'fedmechcl4', 'fedmechclwcl']
    df['method'] = pd.Categorical(df['method'], categories=sort_order, ordered=True)

    # sort the dataframe by the specified column
    df = df.sort_values('method')
    
    return df

if __name__ == "__main__":
    # read arguments from command line
    # scenario = sys.argv[1]
    # mr = sys.argv[2]
    data = "fed_imp10/0716/avila"
    dfs = []
    n_clients = '20'
    sample_size = 0.01
    scenarios = ["mary_lr", "mnar_lr"]
    #scenarios = ["nonignorable_ms_lr"]
    mrs = ["random_in_group2"]
    for scenario in scenarios:
        for mr in mrs:
            df = process_ret(data, scenario, mr, n_clients, sample_size)
            dfs.append(df)

    processed_dir = "./results/processed_results/{}/{}/sample@p={}/".format(data, n_clients, sample_size)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    df = pd.concat(dfs, axis= 0)
    df.to_csv(processed_dir+"result.csv", index=False)

    dfs = []
    n_clients = '40'
    sample_size = 0.025
    scenarios = ["nonignorable_ms_lr"]
    #scenarios = ["nonignorable_ms_lr"]
    mrs = ["random_in_group2"]
    for scenario in scenarios:
        for mr in mrs:
            df = process_ret(data, scenario, mr, n_clients, sample_size)
            dfs.append(df)

    processed_dir = "./results/processed_results/{}/{}/sample@p={}/".format(data, n_clients, sample_size)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    df = pd.concat(dfs, axis= 0)
    df.to_csv(processed_dir+"result.csv", index=False)