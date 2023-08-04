from collections import OrderedDict
from copy import deepcopy
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from src.fed_imp.sub_modules.client.simple_client import SimpleClient
from src.fed_imp.sub_modules.model.logistic import LogisticRegression
from src.fed_imp.sub_modules.model.TwoNN import TwoNN
from src.fed_imp.sub_modules.model.utils import init_net
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
from typing import Dict, List
from loguru import logger
from torch.utils.data import DataLoader, ConcatDataset
from src.utils import set_seed
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredServerFedAvgPytorch:

    def __init__(
            self, clients: Dict[int, SimpleClient], server_config: dict,
            pred_config: dict, test_data: np.ndarray = None, base_model: str = 'twonn'
    ):

        # Basic setup
        self.clients = clients
        self.config = server_config
        self.seed = server_config['seed']
        self.rounds = server_config['pred_rounds']
        self.metric = server_config.get('metric', 'accu')
        self.base_model = base_model

        ###########################################################################
        # Prediction model
        ###########################################################################
        # model
        self.pred_model_params = pred_config.get(
            'model_params', {
                "num_hiddens": 64,
            }
        )

        self.pred_model_params['input_feature_dim'] = test_data.shape[1] - 1
        self.pred_model_params['output_classes_dim'] = len(np.unique(test_data[:, -1]))

        other_params = pred_config["model_params"].get('model_other_params', {})
        if other_params is None:
            other_params = {}

        self.model_init_config = pred_config["model_params"].get('model_init_config', {})
        if self.model_init_config is None:
            self.model_init_config = {}

        # training parameters
        self.pred_training_params = pred_config.get(
            'train_params', {
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "pred_round": 300
            }
        )

        # test data
        self.test_data = test_data
        validate_datas, train_datas = [], []
        for client_id, client in self.clients.items():
            validate_datas.append(client.val_data)
            train_datas.append(client.train_data)

        self.validate_data = np.concatenate(validate_datas, axis=0)
        self.train_data = np.concatenate(train_datas, axis=0)

        self.batch_size = self.pred_training_params['batch_size']
        self.sample_pct = self.pred_training_params['sample_pct']
        self.test_dataset = construct_tensor_dataset(self.test_data[:, :-1], self.test_data[:, -1])
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        self.validate_dataset = construct_tensor_dataset(self.validate_data[:, :-1], self.validate_data[:, -1])
        self.val_dataloader = DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=False)

    def prediction(self):
        ###############################################################################################
        # N rounds Final FedAvg federated prediction model learning
        ###############################################################################################
        # setup work in client side
        seeds = list(range(42, 42 + self.rounds))

        # construct training dataloader
        for client_id, client in self.clients.items():
            client.pred_data_setup(self.batch_size)

        # dataset
        hidden_size = self.pred_model_params['num_hiddens']
        train_epochs = self.pred_training_params['pred_round']
        learning_rate = self.pred_training_params['learning_rate']
        weight_decay = self.pred_training_params['weight_decay']
        local_epoch = self.pred_training_params['local_epoch']

        best_accus, best_f1s, best_rocs, histories = [], [], [], []
        for s in seeds:

            set_seed(s)

            if self.base_model == 'twonn':
                pred_model = TwoNN(
                    in_features=self.train_data.shape[1] - 1, num_classes=len(np.unique(self.train_data[:, -1])),
                    num_hiddens=self.pred_model_params['num_hiddens']
                )
            elif self.base_model == 'lr':
                pred_model = LogisticRegression(
                    in_features=self.train_data.shape[1] - 1, num_classes=len(np.unique(self.train_data[:, -1]))
                )
            else:
                raise ValueError('base model not supported')

            # self.pred_model = init_net(self.pred_model, **self.model_init_config)

            # N epochs of federated training
            clients_prediction_history = []
            best_accu, best_roc, counter, patience = 0, 0, 0, 100
            for current_round in range(1, train_epochs + 1):
                train_loss, test_loss, test_accu, test_f1, test_roc_auc, val_loss, val_accu, val_f1, val_roc_auc \
                    = self._run_round_prediction(
                    pred_model, server_round=current_round, lr=learning_rate, wd=weight_decay,
                    local_epoch=local_epoch
                )

                clients_prediction_history.append(
                    {
                        'test_loss': test_loss, 'test_accu': test_accu, 'test_f1': test_f1, 'test_roc': test_roc_auc,
                        'val_loss': val_loss, 'val_accu': val_accu, 'val_f1': val_f1, 'val_roc': val_roc_auc
                    }
                )

                if current_round % 10 == 0:
                    logger.info(
                        'Round: {}, test_accu: {:.4f}, test_f1: {:.4f}, test_roc: {:.4f}, '
                        'val_loss: {:.4f}, val_accu: {:.4f}, val_f1: {:.4f}'.format(
                            current_round, test_accu, test_f1, test_roc_auc, val_loss, val_accu, val_f1
                        )
                    )

                if current_round >= 100:
                    if self.metric == 'accu':
                        if test_accu > best_accu:
                            best_accu = test_accu
                            counter = 0
                        else:
                            counter += 1

                        if counter >= patience:
                            logger.info("Early stop at round {}".format(current_round))
                            break
                    elif self.metric == 'roc':
                        if test_roc_auc > best_roc:
                            best_roc = test_roc_auc
                            counter = 0
                        else:
                            counter += 1

                        if counter >= patience:
                            logger.info("Early stop at round {}".format(current_round))
                            break

            # Test the model
            accu_rounds_average = [item['test_accu'] for item in clients_prediction_history]
            f1_rounds_average = [item['test_f1'] for item in clients_prediction_history]
            best_accus.append(np.max(accu_rounds_average))
            best_f1s.append(np.max(f1_rounds_average))
            best_rocs.append(np.max([item['test_roc'] for item in clients_prediction_history]))
            histories.append(clients_prediction_history)

        logger.info(
            "model1 test acc: {:.6f} ({:.3f}), test f1: {:.6f} ({:.3f})".format(
                np.array(best_accus).mean(), np.array(best_accus).std(), np.array(best_f1s).mean(),
                np.array(best_f1s).std()
            )
        )

        return {
            "accu_mean": np.array(best_accus).mean(),
            "f1_mean": np.array(best_f1s).mean(),
            'roc_mean': np.array(best_rocs).mean(),
            "accu_std": np.array(best_accus).std(),
            "f1_std": np.array(best_f1s).std(),
            'roc_std': np.array(best_rocs).std(),
            'history': histories,
        }

    ####################################################################################################################
    # Prediction
    ####################################################################################################################
    def _run_round_prediction(self, pred_model, server_round, lr, wd, local_epoch):

        np.random.seed(self.seed + server_round)
        selected_clients_ids = random.sample(self.clients.keys(), k=int(len(self.clients.keys()) * self.sample_pct))

        train_losses = []
        for idx in selected_clients_ids:
            client = self.clients[idx]
            # if init:
            #     torch.manual_seed(self.seed + idx)
            #     pred_model = init_net(pred_model, **self.model_init_config)
            client.pred_model = deepcopy(pred_model)
            train_loss = client.local_train_pred(local_epoch, lr, wd)
            train_losses.append(train_loss)

        # average local training results
        self.average_model(pred_model, selected_clients_ids)

        # evaluation
        val_loss, val_acc, val_f1, val_roc = self.evaluate_pred_model(pred_model, validate=True)
        test_loss, test_acc, test_f1, test_roc = self.evaluate_pred_model(pred_model, validate=False)

        return (
            np.array(train_losses).mean(),
            test_loss, test_acc, test_f1, test_roc,
            val_loss, val_acc, val_f1, val_roc
        )

    def average_model(self, pred_model, selected_clients_ids):

        averaged_weights = OrderedDict()
        clients = [self.clients[idx] for idx in selected_clients_ids]
        sample_sizes = [client.get_sample_size() for client in clients]
        normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]
        for it, client in enumerate(clients):
            local_weights = client.pred_model.state_dict()
            for key in pred_model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = normalized_coefficient[it] * local_weights[key]
                else:
                    averaged_weights[key] += normalized_coefficient[it] * local_weights[key]
        pred_model.load_state_dict(averaged_weights)

    def evaluate_pred_model(self, pred_model, validate=False):
        pred_model.eval()
        pred_model.to(DEVICE)

        if validate:
            test_dataloader = self.val_dataloader
            X_test, y_test = self.validate_data[:, :-1], self.validate_data[:, -1]
        else:
            X_test, y_test = self.test_data[:, :-1], self.test_data[:, -1]
            test_dataloader = self.test_dataloader

        with torch.no_grad():
            test_loss, counter = 0, 0
            for data, labels in test_dataloader:
                counter += 1
                data, labels = data.float().to(DEVICE), labels.long().to(DEVICE)
                outputs = pred_model(data)
                test_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

                # f1 score
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        pred_model.to(DEVICE)
        test_epoch_loss = test_loss / counter
        outputs = pred_model(torch.FloatTensor(X_test).to(DEVICE))
        _, predicted = torch.max(outputs.data, 1)
        probabilities = F.softmax(outputs, dim=1)

        test_accuracy = accuracy_score(y_test, predicted.to('cpu').numpy())
        test_f1 = f1_score(y_test, predicted.to('cpu').numpy(), average='macro')
        if probabilities.shape[1] == 2:
            test_roc_auc = roc_auc_score(
                y_test, probabilities.detach().to('cpu').numpy()[:, 1], average='macro'
            )
        else:
            test_roc_auc = roc_auc_score(
                y_test, probabilities.detach().to('cpu').numpy(), multi_class='ovr', average='macro'
            )

        return test_epoch_loss, test_accuracy, test_f1, test_roc_auc
