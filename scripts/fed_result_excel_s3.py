import os
import json
import pandas as pd
import sys
import numpy as np

import os
type = ''
datasets = ['codon', 'codrna', 'mimic_mo2', 'genetic', 'heart']
scenarios = ["random2@mrl=0.3_mrr=0.7_mm=mnarlrq", "random@mrl=0.3_mrr=0.7_mm=mnarlrq"]
#scenarios = ['s3', 's4']
name = 'random'

############################################################################################################################
# read files
dir_path = './results/raw_results/fed_imp_pc2/1126/'
print(dir_path)
all_dirs, all_files = [], []
for root, dirs, files in os.walk(dir_path):
    for dir in dirs:
        all_dirs.append(os.path.join(root, dir))
    for file in files:
        all_files.append(os.path.join(root, file))

#####################################################################################################
# filter files
filtered_files = []
for file in all_files:
    if file.endswith('.json'):
        needed1 = False
        for keyword in datasets:
            if keyword in file:
                needed1 = True
                break
        needed2 = False
        for keyword in scenarios:
            if keyword in file:
                needed2 = True
                break
        if needed1 and needed2:
            filtered_files.append(file.replace(dir_path, ''))

for file in filtered_files:
    print(file)

print(len(filtered_files))

# def calculate_group_avg(data):
#     rets_central, rets_peri = [], []
#     for key, value in data["results"]["clients_imp_ret_clean"].items():
#         ret = list(value.values())
#         cluster1_client = float(np.array([float(np.array(item[-3:]).mean()) for item in ret[0:]]).mean())
#         cluster2_clients = float(np.array([float(np.array(item[-3:]).mean()) for item in ret[5:]]).mean())
#         rets_central.append(cluster1_client)
#         rets_peri.append(cluster2_clients)
       
    
#     return rets_central, rets_peri
        
datas = []
datas_clients = []
for file in filtered_files:
    with open(os.path.join(dir_path+file), 'r') as fp:
        data = json.load(fp)
        file_record = []
        file_record.extend(file.split('\\'))
        file_record.extend(list(data['results']['avg_imp_final'].values())[0:3] + list(data['results']['avg_imp_final'].values())[3:4])
        
        # calculate rmse for central and peripheries
        # if "central" in data['params']["file_name"]:
        #     file_record.extend([None for _ in range(6)])
        # else:
        #     c_ret, p_ret = calculate_group_avg(data)
        #     file_record.extend(c_ret)
        #     file_record.extend(p_ret)
        
        #file_record.append(data['results']['accu_mean'])
        datas.append(file_record)
        
        file_record_clients = []
        file_record_clients.extend(file.split('\\'))
        clients_result_dict = data['results']["clients_imp_ret_clean"]
        keys = ['imp@rmse', 'imp@sliced_ws']
        for key in keys:
            clients_result = list(clients_result_dict[key].values())[:-1]
            clients_result = [np.array(item[-3:]).mean() for item in clients_result]
            file_record_clients.extend(clients_result)
        datas_clients.append(file_record_clients)
        
print(len(datas))
df = pd.DataFrame(datas)
df2 = pd.DataFrame(datas_clients)
print(df.head())
print(df.shape)
print(df2.head())
print(df2.shape)

output_dir = dir_path.replace('raw_results', 'processed_results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mapping2 = {
    'central': 'central',
    'local': 'local',
    'fedavg-s': 'simpleavg',
    'fedmechw': 'fedmechw'
}

#df = df.applymap(lambda x: mapping1[x] if x in mapping1 else x)
def func(x):
    x = x.split('@')[0]
    x = x[3:]
    for key in mapping2:
        if key in x:
            return x.replace(key, mapping2[key])
    
    return x

df[4] = df[4].apply(lambda x: func(x) if isinstance(x, str) else x)
df2[4] = df2[4].apply(lambda x: func(x) if isinstance(x, str) else x)
df[3] = df[3].apply(lambda x: x.split('@')[0])
df2[3] = df2[3].apply(lambda x: x.split('@')[0])
#
order1 = ["central", 'local', 'simpleavg', 'fedmechw', 'fedmechw_p', 'fedmechw_new']

#df[3] = pd.Categorical(df[3], categories=order2, ordered=True)
df[4] = pd.Categorical(df[4], categories=order1, ordered=True)
df[1] = df[1].astype(int)
columns = ['dataset', 'n_clients', 'sample_size', 'mechanism', 'method', 'rmse', 'ws', 'sliced-ws', 'global-ws']
df.columns = columns
df = df.sort_values(['dataset', 'n_clients', 'mechanism', 'method'])
print(df)

df2[4] = pd.Categorical(df2[4], categories=order1, ordered=True)
df2[1] = df2[1].astype(int)
columns = ['dataset', 'n_clients', 'sample_size', 'mechanism', 'method'] + [f'rmse_c{i}'for i in range(10)] + [f'sliced_ws_c{i}'for i in range(10)]
df2.columns = columns
df2 = df2.sort_values(['dataset', 'n_clients', 'mechanism', 'method'])

# df = df.sort_values([2, 3, 4])
with pd.ExcelWriter(os.path.join(output_dir, 'results{}.xlsx'.format(name))) as writer:
    df.to_excel(writer, index=False, sheet_name = ''.join([item.split('@')[0] for item in scenarios]))
    df2.to_excel(writer, index=False, sheet_name = ''.join([item.split('@')[0] for item in scenarios]) + '_clients')