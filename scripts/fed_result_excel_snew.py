import os
import json
import pandas as pd
import sys
import numpy as np

import os
type = ''
datasets = ['codon', 'codrna', 'mimiciii', 'genetic', 'heart']
#datasets = ['codrna']
#scenarios = ["ideal_perfect", "perfect_s3", "imperfect_s32", "one_side_comp_s33", "no_comp"]
scenarios = ['sample-evenly', 'sample-uneven10range', 'sample-uneven10dir', 'sample-unevenhs']
#scenarios = ["random2@mrl=0.2_mrr=0.8_mm=mnarlrq"]
#scenarios = ['s31', 's32', 's33']
pred = False
exp = 'random'

############################################################################################################################
# read imp and pred files
filtered_files = []
dir_path1 = './results/raw_results/fed_imp_pc2/{}/'.format(exp)
for dataset in datasets:
    for scenario in scenarios:
        path = dir_path1 + dataset + '/' + scenario + '/'
        print(path)
        all_dirs, all_files = [], []
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                all_dirs.append(os.path.join(root, dir))
            for file in files:
                all_files.append(os.path.join(root, file))
        for file in all_files:
            if file.endswith('.json'):
                filtered_files.append(file)

filtered_files_pred = []
dir_path2 = './results/raw_results/fed_imp_pc2_pred_fed/{}/'.format(exp)
for dataset in datasets:
    for scenario in scenarios:
        path = dir_path2 + dataset + '/' + scenario + '/'
        print(path)
        all_dirs, all_files = [], []
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                all_dirs.append(os.path.join(root, dir))
            for file in files:
                all_files.append(os.path.join(root, file))
        for file in all_files:
            if file.endswith('.json'):
                filtered_files_pred.append(file)

#####################################################################################################
datas, datas_pred = [], []
#datas_clients = []
for file in filtered_files:
    print(file)
    with open(file, 'r') as fp:
        data = json.load(fp)
        file_record = []
        file_record.extend(file.split('/')[-3:])
        file_record.extend(list(data['results']['avg_imp_final'].values())[0:3])
        datas.append(file_record)

for file in filtered_files_pred:
    with open(file, 'r') as fp:
        data = json.load(fp)
        file_record = []
        file_record.extend(file.split('/')[-3:])
        file_record.extend(
            [data['results']['accu_mean']*100, data['results']['f1_mean']*100, data['results']['roc_mean']*100, 
             data['results']['prc_mean']*100 if 'prc_mean' in data['results'] else None,
             data['results']["f1_mean_std"]*100, data['results']["roc_mean_std"]*100])
        datas_pred.append(file_record)
        
print(len(datas))
print(len(datas_pred))
df = pd.DataFrame(datas)
df_pred = pd.DataFrame(datas_pred)
print(df.shape)
print(df_pred.shape)
mapping2 = {
    'central': 'central',
    'local': 'local',
    'fedavg-s': 'simpleavg',
    'fedmechw_new': 'fedmechw_new'
}
def func(x):
    x = x.split('@')[0]
    x = x[3:]
    for key in mapping2:
        if key in x:
            return x.replace(key, mapping2[key])
    
    return x

df[2] = df[2].apply(lambda x: func(x) if isinstance(x, str) else x)
df_pred[2] = df_pred[2].apply(lambda x: mapping2[x.replace('_fedavg_mlp_pytorch_pred.json', '')] if isinstance(x, str) else x)

print(df.head())
print(df_pred.head())

df = pd.merge(df, df_pred, how='left', on=[0, 1, 2])
print(df.shape)
output_dir = dir_path1.replace('raw_results', 'processed_results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#df = df.applymap(lambda x: mapping1[x] if x in mapping1 else x)
order1 = ["central", 'local', 'simpleavg', 'fedmechw_new']
print(df.head())
df[2] = pd.Categorical(df[2], categories=order1, ordered=True)
columns = ['dataset', 'scenario', 'method', 'rmse', 'ws', 'sliced-ws', 'accu', 'f1', 'roc', 'prc','f1_std', 'roc_std']
df.columns = columns
df = df.sort_values(['dataset', 'scenario', 'method'])
print(df)

#df2[4] = pd.Categorical(df2[4], categories=order1, ordered=True)
#df2[1] = df2[1].astype(int)
#columns = ['dataset', 'n_clients', 'sample_size', 'mechanism', 'method'] + [f'rmse_c{i}'for i in range(10)] + [f'sliced_ws_c{i}'for i in range(10)]
#df2.columns = columns
#df2 = df2.sort_values(['dataset', 'n_clients', 'mechanism', 'method'])

#####################################################################################################
# if pred:
#     pred_dir_path = './results/raw_results/fed_imp_pc2_pred_fed/new/'
#     all_pred_dirs, all_pred_files = [], []
#     for root, dirs, files in os.walk(pred_dir_path):
#         for dir in dirs:
#             all_pred_dirs.append(os.path.join(root, dir))
#         for file in files:
#             all_pred_files.append(os.path.join(root, file))

#     filtered_files_pred = []
#     for file in all_pred_files:
#         if file.endswith('.json'):
#             needed1 = False
#             for keyword in datasets:
#                 if keyword in file:
#                     needed1 = True
#                     break
#             needed2 = False
#             for keyword in scenarios:
#                 if keyword in file:
#                     needed2 = True
#                     break
#             if needed1 and needed2:
#                 filtered_files_pred.append(file.replace(pred_dir_path, ''))

#     datas = []
#     datas_clients = []
#     for file in filtered_files_pred:
#         with open(os.path.join(pred_dir_path+file), 'r') as fp:
#             data = json.load(fp)
#             file_record = []
#             file_record.extend(file.split('\\'))
#             file_record.extend([data['results']['accu_mean']*100, data['results']['f1_mean']*100, data['results']['roc_mean']*100, data['results']['prc_mean']*100 if 'prc_mean' in data['results'] else None])
            
#             # calculate rmse for central and peripheries
#             # if "central" in data['params']["file_name"]:
#             #     file_record.extend([None for _ in range(6)])
#             # else:
#             #     c_ret, p_ret = calculate_group_avg(data)
#             #     file_record.extend(c_ret)
#             #     file_record.extend(p_ret)
            
#             #file_record.append(data['results']['accu_mean'])
#             datas.append(file_record)
#     df_pred = pd.DataFrame(datas)
#     print(df_pred[4])
#     def func1(x):
#         x = x.replace("_fedavg_mlp_pytorch_pred.json", "")
#         for key in mapping2:
#             if key in x:
#                 return x.replace(key, mapping2[key])
#         return x
#     df_pred[4] = df_pred[4].apply(lambda x: func1(x) if isinstance(x, str) else x)
#     df_pred[3] = df_pred[3].apply(lambda x: x.split('@')[0])
#     df_pred[4] = pd.Categorical(df_pred[4], categories=order1, ordered=True)
#     df_pred[1] = df_pred[1].astype(int)
#     columns = ['dataset', 'n_clients', 'sample_size', 'mechanism', 'method', 'accu', 'f1', 'roc', 'prc']
#     df_pred.columns = columns
#     df_pred = df_pred.sort_values(['dataset', 'n_clients', 'mechanism', 'method'])
#     print(df_pred)

# df = df.sort_values([2, 3, 4])
for dataset in datasets:
    df_new = df[df['dataset'] == dataset]
    with pd.ExcelWriter(os.path.join(output_dir, '{}.xlsx'.format(dataset))) as writer:
        df_new.to_excel(writer, index=False, sheet_name = 'imp')
    #df2.to_excel(writer, index=False, sheet_name = ''.join([item.split('@')[0] for item in scenarios]) + '_clients')
    #if pred:
        #df_pred.to_excel(writer, index=False, sheet_name = ''.join([item.split('@')[0] for item in scenarios]) + '_pred')