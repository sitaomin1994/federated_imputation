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

def main_prediction(data_dir, server_name, server_config, server_pred_config, test_data, n_clients, round_):
    print("round: {}".format(round_))
    data_imp = np.load(os.path.join(data_dir, "imputed_data_{}.npy".format(round_)))
    data_true = np.load(os.path.join(data_dir, "origin_data_{}.npy".format(round_)))
    missing_mask = np.load(os.path.join(data_dir, "missing_mask_{}.npy".format(round_)))
    # print("data_imp.shape: {}".format(data_imp.shape))
    # print("data_true.shape: {}".format(data_true.shape))
    # print("missing_mask.shape: {}".format(missing_mask.shape))

    # setup client
    clients = {}
    for client_id in range(n_clients):
        clients[client_id] = SimpleClient(
            client_id=client_id,
            data_imp=data_imp[client_id],
            missing_mask=missing_mask[client_id],
            data_true=data_true[client_id],
            data_test=test_data.values
        )

    # setup server
    server = load_server(
        server_name, clients=clients, server_config=server_config, pred_config=server_pred_config,
        test_data=test_data.values
    )

    # prediction
    ret = server.prediction()
    return ret


def prediction(main_config, server_config_, pred_rounds, seed, mtp=False):
    data = main_config["data"]
    n_clients = main_config["n_clients"]
    sample_size = main_config["sample_size"]
    scenario = main_config["scenario"]
    mr = main_config["mr"]
    method = main_config["method"]
    # server
    server_name = server_config_["server_name"]
    server_pred_config = server_config_["server_pred_config"]
    server_config = server_config_["server_config"]
    server_config["pred_rounds"] = pred_rounds
    server_config["seed"] = seed

    ###################################################################################
    # Main part
    ###################################################################################
    root_dir = "./results/raw_results/{}/{}/sample@p={}/{}/{}/".format(data, n_clients, sample_size, scenario, mr)
    data_dir, exp_file = get_all_dirs(root_dir, method)

    ####################################################################################
    # Find overall train and test data
    ####################################################################################
    with open(os.path.join(exp_file), 'r') as fp:
        exp_ret = json.load(fp)
    exp_config = exp_ret["params"]["config"]

    dataset_params = exp_config['data']
    data, data_config = load_data(**dataset_params)
    seed = exp_config['experiment']['seed']
    n_rounds = exp_config['experiment']['n_rounds']
    if n_rounds == 1:
        n_rounds_data = split_train_test(data, n_folds=2, seed=seed)
    else:
        n_rounds_data = split_train_test(data, n_folds=n_rounds, seed=seed)

    # n rounds average
    train_data, test_data = n_rounds_data[0]
    print("train_data.shape: {}".format(train_data.shape))
    print("test_data.shape: {}".format(test_data.shape))

    ####################################################################################
    # load clients imputed and original datas
    ####################################################################################
    print("==============================================================")
    print("n_rounds: {}".format(n_rounds))

    rets = []
    n_rounds = 1
    if mtp == False:
        for round_ in range(1, n_rounds + 1):
            print("round: {}".format(round_))
            data_imp = np.load(os.path.join(data_dir, "imputed_data_{}.npy".format(round_)))
            data_true = np.load(os.path.join(data_dir, "origin_data_{}.npy".format(round_)))
            missing_mask = np.load(os.path.join(data_dir, "missing_mask_{}.npy".format(round_)))
            # print("data_imp.shape: {}".format(data_imp.shape))
            # print("data_true.shape: {}".format(data_true.shape))
            # print("missing_mask.shape: {}".format(missing_mask.shape))
            # setup client
            clients = {}
            for client_id in range(n_clients):
                clients[client_id] = SimpleClient(
                    client_id=client_id,
                    data_imp=data_imp[client_id],
                    missing_mask=missing_mask[client_id],
                    data_true=data_true[client_id],
                    data_test=test_data.values
                )

            # setup server
            server = load_server(
                server_name, clients=clients, server_config=server_config, pred_config=server_pred_config,
                test_data=test_data.values
            )

            # prediction
            ret = server.prediction()
            rets.append(ret)
    else:
        n_process = n_rounds
        chunk_size = n_rounds // n_process
        rounds = list(range(n_rounds))

        with mp.Pool(n_process) as pool:
            process_args = [
                (data_dir, server_name, server_config, server_pred_config, test_data, n_clients, round_)
                for round_ in rounds]
            process_results = pool.starmap(main_prediction, process_args, chunksize=chunk_size)

        rets = process_results

    # average results
    average_ret = {}
    for key in rets[0].keys():
        if key != 'history':
            average_ret[key] = np.mean([ret[key] for ret in rets])
            average_ret['{}_std'.format(key)] = np.std([ret[key] for ret in rets])

    print(average_ret)
    items = root_dir.split('/')
    new_items = []
    for item in items:
        if 'fed_imp' in item:
            new_items.append(item + '_pred_fed')
        else:
            new_items.append(item)
    pred_result_dir = '/'.join(new_items)

    if not os.path.exists(pred_result_dir):
        os.makedirs(pred_result_dir)

    print(pred_result_dir)

    pred_exp_filepath = pred_result_dir + '{}_{}.json'.format(main_config['method'], server_config_['server_name'])
    print(pred_exp_filepath)

    pred_exp_ret_content = {
        'params': {
            "main_config": main_config,
            "server_config": server_config_,
        },
        'results': average_ret,
        "raw_results": rets
    }
    with open(pred_exp_filepath, 'w') as fp:
        json.dump(pred_exp_ret_content, fp)

    return average_ret


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

    for file in all_files:
        if method in file and file.endswith(".json"):
            exp_file = file

    if data_dir is None or exp_file is None:
        raise ValueError("No folder for such method: {}".format(method))
    else:
        return data_dir, exp_file


if __name__ == '__main__':
    main_config_tmpl = {
        "data": "fed_imp12/0717/ijcnn_balanced",
        "n_clients": 20,
        "sample_size": 0.01,
        "scenario": "mary_lr",
        "mr": "random_in_group2",
        "method": "fedavg-s"
    }

    server_config_tmpl = {
        "server_name": 'central_mlp_pytorch_pred',
        "server_pred_config": {
            "model_params": {
                "model": "2nn",
                "num_hiddens": 32,
                "model_init_config": None,
                "model_other_params": None
            },
            "train_params": {
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.001,
                "pred_round": 1500,
                "pred_local_epochs": 3,
                'local_epoch': 5,
                'sample_pct': 1
            }
        },
        "server_config": {
            'pred_rounds': 3,
            'seed': 21
        }
    }

    pred_rounds = 1
    seed = 21
    mtp = False

    datasets = [
        #('fed_imp10/0716/ijcnn_balanced', 0.01, 0.01),
        #('fed_imp12/0717/svm_g', 0.01, 0.01),
        # ('fed_imp10/0716/adult', 0.01, 0.01),
        # ('fed_imp10/0716/default_credit', 0.05, 0.05),
        #('bank_balanced_pca', 0.01, 0.01),
        #('fed_imp10/0716/bank_marketing_balanced', 0.05, 0.05),
        #('fed_imp10/0716/statlog', 0.05, 0.05),
        #('fed_imp12/0717/bank_marketing_balanced', 0.01, 0.01),
        # ('fed_imp10/0716/susy', 0.01, 0.01)
        # ('fed_imp10/0716/sensor', 0.01, 0.01),
        #('skin_balanced', 0.005, 0.005),
        # ('statlog' , 0.05, 0.05),
        #('avila', 0.01, 0.025),
        # ('fed_imp12/0717/pendigits', 0.01, 0.025),
        # ('fed_imp10/0716/svm', 0.05, 0.05),
        ('fed_imp10/0716/susy', 0.01, 0.01)
    ]

    servers = ['fedavg_mlp_pytorch_pred']
    methods = ['local', 'fedavg-s', 'fedmechw'] # 'fedmechw'

    for dataset, ss1, ss2 in datasets:
        for server in servers:
            #mary
            for method in methods:
                logger.info(
                    "===============================================================================================")
                logger.info("dataset: {}, server: {}, method: {} mm: mary".format(dataset, server, method))
                main_config = copy.deepcopy(main_config_tmpl)
                main_config['data'] = dataset
                main_config['sample_size'] = ss1
                main_config['scenario'] = 'mary_lr'
                main_config['method'] = method
                server_config = copy.deepcopy(server_config_tmpl)
                server_config['server_name'] = server

                prediction(main_config, server_config, pred_rounds, seed, mtp = mtp)

            # mnar
            for method in methods:
                logger.info(
                    "===============================================================================================")
                logger.info("dataset: {}, server: {}, method: {} mm: mnar".format(dataset, server, method))
                main_config = copy.deepcopy(main_config_tmpl)
                main_config['data'] = dataset
                main_config['sample_size'] = ss1
                main_config['scenario'] = 'mnar_lr'
                main_config['method'] = method
                server_config = copy.deepcopy(server_config_tmpl)
                server_config['server_name'] = server

                prediction(main_config, server_config, pred_rounds, seed, mtp = mtp)

            #nonignorable_ms_lr
            for method in methods:
                logger.info(
                    "===============================================================================================")
                logger.info("dataset: {}, server: {}, method: {} mm: mixed".format(dataset, server, method))
                main_config = copy.deepcopy(main_config_tmpl)
                main_config['data'] = dataset
                main_config['n_clients'] = 40
                main_config['sample_size'] = ss2
                main_config['scenario'] = 'nonignorable_ms_lr'
                main_config['method'] = method
                server_config = copy.deepcopy(server_config_tmpl)
                server_config['server_name'] = server

                prediction(main_config, server_config, pred_rounds, seed, mtp=mtp)