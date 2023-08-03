import os
import json
import pandas as pd
import sys

# def process_ret(data, scenario, mr, n_clients, sample_size):
#     dir_path = "./results/raw_results/{}/{}/sample@p={}/{}/{}/".format(data, n_clients, sample_size, scenario, mr)
#     # Read all json files from directory
#     files = os.listdir(dir_path)
#     files = [f for f in files if f.endswith('.json')]
#     # Read all json files
#     df_data = []
#     for f in files:
#         with open(dir_path+f, 'r') as fp:
#             ret = {}
#             data = json.load(fp)
#             ret['scenario'] = scenario
#             ret['mr'] = mr
#             ret['method'] = data["params"]["config"]["agg_strategy_imp"]["strategy"]
#             ret['rmse'] = data["results"]["avg_imp_final"]["imp@rmse"]
#             ret['ws'] = data["results"]["avg_imp_final"]["imp@w2"]
#             ret['sliced_ws'] = data["results"]["avg_imp_final"]["imp@sliced_ws"]
#             # ret['fed_accu'] = data["results"]["avg_pred_final_model1"]["accu"]
#             # ret['fed_f1'] = data["results"]["avg_pred_final_model1"]["f1"]
#             # ret['fed_accu_std'] = data["results"]["avg_pred_final_model1"]["accu_std"]
#             # ret['fed_f1_std'] = data["results"]["avg_pred_final_model1"]["f1_std"]
#             df_data.append(ret)
    
#     df = pd.DataFrame(df_data)
#     sort_order = ['local', 'fedavg', 'fedavg-s', 'fedwavg', 'fedmechw', 'fedmechclw', 'fedwavgcl', 'fedmechcl4', 'fedmechclwcl']
#     df['method'] = pd.Categorical(df['method'], categories=sort_order, ordered=True)

#     # sort the dataframe by the specified column
#     df = df.sort_values('method')
    
#     return df

# if __name__ == "__main__":
#     # read arguments from command line
#     # scenario = sys.argv[1]
#     # mr = sys.argv[2]
#     data = "fed_imp10/0716/avila"
#     dfs = []
#     n_clients = '20'
#     sample_size = 0.01
#     scenarios = ["mary_lr", "mnar_lr"]
#     #scenarios = ["nonignorable_ms_lr"]
#     mrs = ["random_in_group2"]
#     for scenario in scenarios:
#         for mr in mrs:
#             df = process_ret(data, scenario, mr, n_clients, sample_size)
#             dfs.append(df)

#     processed_dir = "./results/processed_results/{}/{}/sample@p={}/".format(data, n_clients, sample_size)
#     if not os.path.exists(processed_dir):
#         os.makedirs(processed_dir)
    
#     df = pd.concat(dfs, axis= 0)
#     df.to_csv(processed_dir+"result.csv", index=False)

#     dfs = []
#     n_clients = '40'
#     sample_size = 0.025
#     scenarios = ["nonignorable_ms_lr"]
#     #scenarios = ["nonignorable_ms_lr"]
#     mrs = ["random_in_group2"]
#     for scenario in scenarios:
#         for mr in mrs:
#             df = process_ret(data, scenario, mr, n_clients, sample_size)
#             dfs.append(df)

#     processed_dir = "./results/processed_results/{}/{}/sample@p={}/".format(data, n_clients, sample_size)
#     if not os.path.exists(processed_dir):
#         os.makedirs(processed_dir)
    
#     df = pd.concat(dfs, axis= 0)
#     df.to_csv(processed_dir+"result.csv", index=False)

import os
type = ''
dir_path = './results/raw_results/fed_imp_pc2{}/0727/nhis_income_pca/'.format(type)
all_dirs, all_files = [], []
for root, dirs, files in os.walk(dir_path):
    for dir in dirs:
        all_dirs.append(os.path.join(root, dir))
    for file in files:
        all_files.append(os.path.join(root, file))

filtered_files = []
for file in all_files:
    if file.endswith('.json'):
        filtered_files.append(file.replace(dir_path, ''))

for file in filtered_files:
    print(file)

print(len(filtered_files))
datas = []

for file in filtered_files:
    with open(os.path.join(dir_path+file), 'r') as fp:
        data = json.load(fp)
        file_record = []
        file_record.extend(file.split('\\'))
        file_record.extend(list(data['results']['avg_imp_final'].values())[0:3])
        #file_record.append(data['results']['accu_mean'])
        datas.append(file_record)

print(len(datas))

df = pd.DataFrame(datas)
print(df.head())
print(df.shape)

output_dir = dir_path.replace('raw_results', 'processed_results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# post processing
mapping1 = {
    'fixed@mr=0.1': '0.1 0.1',
    'fixed@mr=0.3': '0.3 0.3',
    'fixed@mr=0.5': '0.5 0.5',
    'fixed@mr=0.7': '0.7 0.7',
    'compl@mr=0.1': '0.1 0.9',
    'compl@mr=0.3': '0.3 0.7',
    'compl@mr=0.7': '0.7 0.3',
}

mapping2 = {
    'local': 'local',
    'fedavg-s': 'simpleavg',
    'fedmechw': 'fedmechw',
}


df = df.applymap(lambda x: mapping1[x] if x in mapping1 else x)

def func(x):
    for key in mapping2:
        if key in x:
            return mapping2[key]
    
    return x
df = df.applymap(lambda x: func(x) if isinstance(x, str) else x)
# df[2] = df[2].apply(lambda x: x.split('=')[-1])
#
# order1 = ['local', 'simpleavg', 'fedmechw']
# order2 = ['0.1 0.1', '0.3 0.3', '0.3 0.7', '0.5 0.5', '0.7 0.7', '0.7 0.3', '0.1 0.9']
#
# df[3] = pd.Categorical(df[3], categories=order2, ordered=True)
# df[4] = pd.Categorical(df[4], categories=order1, ordered=True)
# df = df.sort_values([2, 3, 4])
with pd.ExcelWriter(os.path.join(output_dir, 'result{}.xlsx'.format(type))) as writer:
    for value in df[2].unique():
        df[df[2] == value].to_excel(writer, index=False, sheet_name = 'sp_{}'.format(value)) 