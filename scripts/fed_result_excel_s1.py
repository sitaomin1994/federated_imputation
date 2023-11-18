import os
import json
import pandas as pd
import sys
import numpy as np
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
dir_path = './results/raw_results/fed_imp_pc2{}/1124/codon/'.format(type)
print(dir_path)
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

def calculate_group_avg(data, mech):
    rets_central, rets_peri = [], []
    for key, value in data["results"]["clients_imp_ret_clean"].items():
        if mech == 'l1':
            ret = list(value.values())
            central_client = float(np.array(ret[0][-3:]).mean())
            peripheral_clients = float(np.array([float(np.array(item[-3:]).mean()) for item in ret[1:]]).mean())
            rets_central.append(central_client)
            rets_peri.append(peripheral_clients)
        elif mech == 'r1':
            ret = list(value.values())
            central_client = float(np.array(ret[-1][-3:]).mean())
            peripheral_clients = float(np.array([float(np.array(item[-3:]).mean()) for item in ret[0:-1]]).mean())
            rets_central.append(central_client)
            rets_peri.append(peripheral_clients)
        else:
            raise ValueError('mech not supported')
    
    return rets_central, rets_peri
        

for file in filtered_files:
    with open(os.path.join(dir_path+file), 'r') as fp:
        data = json.load(fp)
        file_record = []
        file_record.extend(file.split('\\'))
        file_record.extend(list(data['results']['avg_imp_final'].values())[0:3])
        # calculate rmse for central and peripheries
        if "l1" in data['params']["dir_name"]:
            if "central" in data['params']["file_name"]:
                file_record.extend([None for _ in range(6)])
            else:
                c_ret, p_ret = calculate_group_avg(data, 'l1')
                file_record.extend(c_ret)
                file_record.extend(p_ret)
                
        elif "r1" in data["params"]["dir_name"]:
            if "central" in data['params']["file_name"]:
                file_record.extend([None for _ in range(6)])
            else:
                c_ret, p_ret = calculate_group_avg(data, 'l1')
                file_record.extend(c_ret)
                file_record.extend(p_ret)
        
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
mapping2 = {
    'central': 'central',
    'local': 'local',
    'fedavg-s': 'simpleavg',
    'fedmechw': 'fedmechw'
}


def func(x):
    x = x.split('@')[0]
    x = x[3:]
    for key in mapping2:
        if key in x:
            return x.replace(key, mapping2[key])
    
    return x
df[4] = df[4].apply(lambda x: func(x) if isinstance(x, str) else x)
# df[2] = df[2].apply(lambda x: x.split('=')[-1])
#
order1 = ["central", 'local', 'simpleavg', 'fedmechw', 'fedmechw_p', 'fedmechw_new']
# order2 = ['0.1 0.1', '0.3 0.3', '0.3 0.7', '0.5 0.5', '0.7 0.7', '0.7 0.3', '0.1 0.9']
#
# df[3] = pd.Categorical(df[3], categories=order2, ordered=True)
df[4] = pd.Categorical(df[4], categories=order1, ordered=True)
df[0] = df[0].astype(int)

df2 = df[(df[0] != 10) & (df[2] == 'mnar_lr@sp=extremer1')].copy()
df2.sort_values([0, 4], inplace=True)

df3 = df[(df[0] != 10) & (df[2] == 'mnar_lr@sp=extremel1')].copy()
df3.sort_values([0, 4], inplace=True)

columns = ['n_clients', 'sample_size', 'mechanism', 'mr', 'method', 'rmse', 'ws', 'sliced-ws', 
        'cent_rmse', 'cent_ws', 'cent_sliced-ws', 'peri_rmse', 'peri_ws', 'peri_sliced-ws'
        ]
df2.columns = columns
df3.columns = columns
# df = df.sort_values([2, 3, 4])
print(df2.head())
print(df3.head())
with pd.ExcelWriter(os.path.join(output_dir, 'result{}.xlsx'.format(type))) as writer:
    df2.to_excel(writer, index=False, sheet_name = 'exp1-lr')
    df3.to_excel(writer, index=False, sheet_name = 'exp1-rl')
    # for value in df[2].unique():
    #     df[df[2] == value].to_excel(writer, index=False, sheet_name = 'sp_{}'.format(value)) 