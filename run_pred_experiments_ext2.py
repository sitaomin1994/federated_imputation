import os
import numpy as np
import json
from src.fed_imp.sub_modules.client.simple_client import SimpleClient
from src.modules.data_preprocessing import load_data
from src.modules.data_spliting import split_train_test
from src.fed_imp.sub_modules.server.load_server import load_server
import multiprocessing as mp
import copy
from loguru import logger
from src.modules.evaluation.imputation_quality import rmse


def main_prediction(data_dir, server_name, server_config, server_pred_config, round_, imbalance):

    data_imp = np.load(os.path.join(data_dir, "imputed_data_{}.npy".format(round_)))
    data_true = np.load(os.path.join(data_dir, "origin_data_{}.npy".format(round_)))
    missing_mask = np.load(os.path.join(data_dir, "missing_mask_{}.npy".format(round_)))
    test_data = np.load(os.path.join(data_dir, "test_data_{}.npy".format(round_)))
    sp = np.load(os.path.join(data_dir, "split_indices_{}.npy".format(round_)))
    data_imps = np.split(data_imp, sp, axis=0)
    data_trues = np.split(data_true, sp, axis=0)
    missing_masks = np.split(missing_mask, sp, axis=0)
    
    # for data_imp, data_true, missing_mask in zip(data_imps, data_trues, missing_masks):
    #     print(data_dir, rmse(data_imp[:, :-1], data_true[:, :-1], missing_mask))

    # setup client
    clients = {}
    for client_id in range(len(data_imps)):
        clients[client_id] = SimpleClient(
            client_id=client_id,
            data_imp=data_imps[client_id],
            missing_mask=missing_masks[client_id],
            data_true=data_trues[client_id],
            data_test=test_data,
            imbalance = imbalance,
            regression=server_pred_config['regression']
        )

    # setup server
    server = load_server(
        server_name, clients=clients, server_config=server_config, pred_config=server_pred_config,
        test_data=test_data, base_model=server_pred_config["model_params"]['model'],
        regression=server_pred_config['regression']
    )

    # prediction
    ret = server.prediction()
    return ret


def prediction(main_config, server_config_, pred_rounds, seed):
    
    # main config
    dataname = main_config["data"]
    n_clients = main_config["n_clients"]
    sample_size = main_config["sample_size"]
    mr_strategy = main_config["mr"]
    method = main_config["method"]
    imbalance = main_config["imbalance"]
    round_idx = main_config["round_idx"]

    # server
    server_name = server_config_["server_name"]
    server_pred_config = server_config_["server_pred_config"]
    server_config = server_config_["server_config"]
    server_config["pred_rounds"] = pred_rounds
    server_config["seed"] = seed

    ###################################################################################
    # Main part
    ###################################################################################
    root_dir = f"./results/raw_results/{dataname}/{n_clients}/{sample_size}/{mr_strategy}/allk0.25_sphere/"

    print(root_dir)
    data_dir, exp_file = get_all_dirs(root_dir, method)

    ####################################################################################
    # Find overall train and test data
    ####################################################################################
    print("====================================================================================")
    round_ = round_idx
    
    print("round: {}".format(round_))
    ret = main_prediction(data_dir, server_name, server_config, server_pred_config, round_, imbalance)
        
    return ret


def get_all_dirs(root_dir, method):
    all_dirs, all_files = [], []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            all_dirs.append(os.path.join(root, dir))
        for file in files:
            all_files.append(os.path.join(root, file))
    data_dir, exp_file = None, None
    for dir_ in all_dirs:
        if method in dir_:
            data_dir = dir_
    #print(all_dirs, all_files)
    for file in all_files:
        if method in file and file.endswith(".json"):
            exp_file = file

    if data_dir is None or exp_file is None:
        raise ValueError("No folder for such method: {}".format(method))
    else:
        return data_dir, exp_file

def parse_comma_separated(s):
    if not s:
        return []
    return [item.strip() for item in s.split(',') if item.strip()]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtp', type=bool, default=True)
    parser.add_argument('--dir_name', type=str, default='fed_imp_ext2_pc2')
    parser.add_argument(
        '--datasets', 
        type=parse_comma_separated,
        help='Comma-separated list of datasets (e.g., data1,data2,data3)',
        default=['1105/hhp_los_np1']
    )
    parser.add_argument(
        '--tasks',
        type=parse_comma_separated,
        help='Comma-separated list of tasks (e.g., task1,task2,task3) reg, cls',
        default=['reg']
    )
    parser.add_argument(
        '--mechanism',
        type=parse_comma_separated,
        help='Comma-separated list of mechanisms (e.g., mechanism1,mechanism2,mechanism3)',
        default=['random2@mrl=0.3_mrr=0.7_mm=mnarlrq']
    )
    parser.add_argument(
        '--methods',
        type=parse_comma_separated,
        help='Comma-separated list of methods (e.g., method1,method2,method3)',
        default=['fedavg-s', 'fedmechw_new', 'local']
    )
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--num_hiddens', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_jobs', type=int, default=-1)  # number of parallel jobs
    return parser.parse_args()

def process_single_task(
    dir_name, d, t, m, method, round_idx, train_param, main_config_tmpl, server_config_tmpl, pred_rounds, seed
):
    dataset = f'{dir_name}/{d}'
    
    main_config = copy.deepcopy(main_config_tmpl)
    main_config['data'] = dataset
    main_config['n_clients'] = 0
    main_config['sample_size'] = 'natural'
    main_config['imbalance'] = None
    main_config['mr'] = m
    main_config['round_idx'] = round_idx
    main_config['method'] = method
    
    server_config = copy.deepcopy(server_config_tmpl)
    server_config["server_pred_config"]["model_params"]["num_hiddens"] = train_param["num_hiddens"]
    server_config["server_pred_config"]["train_params"]["batch_size"] = train_param["batch_size"]
    server_config["server_pred_config"]["train_params"]["learning_rate"] = train_param["lr"]
    server_config["server_pred_config"]["train_params"]["weight_decay"] = train_param["weight_decay"]
    server_config['server_name'] = 'fedavg_mlp_pytorch_pred'
    
    if t == 'reg':
        server_config["server_pred_config"]["regression"] = True
        server_config["server_pred_config"]["model_params"]["model"] = 'twonn_reg'
    else:
        server_config["server_pred_config"]["regression"] = False
        server_config["server_pred_config"]["model_params"]["model"] = 'twonn'
    
    ret = prediction(main_config, server_config, pred_rounds, seed)
    return (d, m, method, round_idx), ret


if __name__ == '__main__':
    import argparse
    from joblib import Parallel, delayed

    
    main_config_tmpl = {
        "data": "fed_imp12/0717/ijcnn_balanced",
        "n_clients": 0,
        "sample_size": "natural",
        "mr": 'random2@mrl=0.3_mrr=0.7_mm=mnarlq',
        "method": "fedavg-s",
        "round_idx": 0,
        "imbalance": None
    }

    server_config_tmpl = {
        "server_name": 'fedavg_mlp_pytorch_pred',
        "server_pred_config": {
            "model_params": {
                "model": "twonn",
                "num_hiddens": 32,
                "model_init_config": None,
                "model_other_params": None
            },
            "train_params": {
                "batch_size": 256,
                "learning_rate": 0.001,
                "weight_decay": 0.001,
                "pred_round": 600,
                "pred_local_epochs": 3,
                'local_epoch': 5,
                'sample_pct': 1
            },
            "regression": False,
        },
        "server_config": {
            'pred_rounds': 1,
            'seed': 21,
            "metric": "f1"
        }
    }
    
    ####################################################################################
    # Parse arguments
    # Example:
    #    python run_pred_experiments_ext2.py 
    #           --dir_name fed_imp_ext2_pc2 --datasets 1105/hhp_los_np1 
    #           --mechanism random2@mrl=0.3_mrr=0.7_mm=mnarlq 
    #           --methods fedavg-s,fedmechw_new,local --n_rounds 5
    #           --num_hiddens 32 --batch_size 300 --lr 0.001 --weight_decay 0.0 --n_jobs -1 --mtp
    ####################################################################################
    """
    python run_pred_experiments_ext2.py --dir_name fed_imp_ext2_pc2 --datasets 1105/hhp_los_np1 --tasks reg --mechanism random2@mrl=0.3_mrr=0.7_mm=mnarlrq --methods fedavg-s,fedmechw_new,local --n_rounds 5 --num_hiddens 32 --batch_size 300 --lr 0.001 --weight_decay 0.0 --n_jobs -1 --mtp True
    """
    args = parse_args()
    pred_rounds = 1
    seed = 21
    mtp = args.mtp
    dir_name = args.dir_name
    
    train_param = {
        "num_hiddens": args.num_hiddens, 
        "batch_size": args.batch_size, 
        "lr": args.lr, 
        "weight_decay": args.weight_decay, 
        'imbalance': None
    }
    
    tasks = [
        (d, t, m, method, round_idx)
        for d,t in zip(args.datasets, args.tasks)
        for m in args.mechanism
        for method in args.methods
        for round_idx in range(args.n_rounds)
    ]
        
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_single_task)(
            dir_name, d, t, m, method, round_idx, train_param, main_config_tmpl, server_config_tmpl, pred_rounds, seed, 
        )
        for d, t, m, method, round_idx in tasks
    )
    
    # Convert results to dictionary
    rets = dict(results)
                    
    ####################################################################################
    # Merge results
    pred_dir_name = dir_name + '_pred_fed'
    for d in args.datasets:
        for m in args.mechanism:
            for method in args.methods:
                
                # all rounds results
                all_rounds_rets = [rets[(d, m, method, round_idx)] for round_idx in range(args.n_rounds)]
                
                # average results
                average_ret = {}
                for key in all_rounds_rets[0].keys():
                    if key != 'history':
                        average_ret[key] = np.mean([ret[key] for ret in all_rounds_rets])
                        average_ret['{}_std'.format(key)] = np.std([ret[key] for ret in all_rounds_rets])

                print(average_ret)
                
                # save_directory
                pred_result_dir = f"./results/raw_results/{pred_dir_name}/{d}/{m}/"

                if not os.path.exists(pred_result_dir):
                    os.makedirs(pred_result_dir)

                pred_exp_filepath = pred_result_dir + '{}.json'.format(method)
                print(pred_exp_filepath)
                
                # save results
                server_config = copy.deepcopy(server_config_tmpl)
                server_config["server_pred_config"]["model_params"]["num_hiddens"] = train_param["num_hiddens"]
                server_config["server_pred_config"]["train_params"]["batch_size"] = train_param["batch_size"]
                server_config["server_pred_config"]["train_params"]["learning_rate"] = train_param["lr"]
                server_config["server_pred_config"]["train_params"]["weight_decay"] = train_param["weight_decay"]

                pred_exp_ret_content = {
                    'params': {
                        "server_config": server_config,
                        'dataset': d,
                        'mechanism': m,
                        'method': method
                    },
                    'results': average_ret,
                    "raw_results": all_rounds_rets
                }
                with open(pred_exp_filepath, 'w') as fp:
                    json.dump(pred_exp_ret_content, fp)
                    
                    