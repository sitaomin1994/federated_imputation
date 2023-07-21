import numpy as np
from sklearn.impute import IterativeImputer, SimpleImputer
from src.modules.data_preprocessing import load_data
from .sub_modules.data_spliting import split_train_test, partition_data
from src.missing_data_exploration.sub_modules.ms_simulate.ms_simulate_hy import simulate_nan
from src.modules.iterative_imputation.local_imputer import LocalIterativeImputer
from src.modules.evaluation.evaluation import Evaluator
from loguru import logger


class ExperimentMSEvalPartition:

	name = 'imp_eval_partition'

	def __init__(self):
		pass

	def run_experiment(self, configuration):

		# Params
		ms_ratio = configuration['missing_simulate']['ms_ratio']
		ms_features = configuration['missing_simulate']['ms_features']
		ms_mechanism = configuration['missing_simulate']['ms_mechanism']
		clf_name = configuration['clf_model']['name']
		clf_tune_params = configuration['clf_model']['tune_params']
		imputer_name = configuration['imputation']['name']

		# load dataset
		dataset_name = configuration['dataset']['name']
		data, data_config = load_data(dataset_name)

		# n_rounds splitting
		n_splits = configuration['fed_imp']['n_rounds']
		seed = configuration['fed_imp']['seed']
		train_test_sets = split_train_test(data, n_splits, seed)

		eval_ret_all = []
		for idx, (train_set, test_set) in enumerate(train_test_sets):

			# pre setup
			X_train, y_train = train_set.values[:, :-1], train_set.values[:, -1]
			X_test, y_test = test_set.values[:, :-1], test_set.values[:, -1]
			logger.info("Round - {}".format(idx))

			# client partition
			k = configuration['data_partition']['k']
			datasets = partition_data(train_set.values, k)

			# debug
			for data in datasets:
				logger.debug(data.shape)

			# imputation for each client
			datasets_filled, masks = [], []
			for data in datasets:

				# missing simulate
				if ms_ratio != 0.0:
					ret = simulate_nan(
						data, ms_ratio, ms_mechanism, p_na_cols=ms_features,
						seed=(seed + idx * 1000809) % 1000009890977
					)

					data_ms = ret['train_data_ms']
					masks.append(np.isnan(data_ms))

					# imputation
					if imputer_name == 'iterative-local':
						imp = LocalIterativeImputer(
							max_iter=configuration["imputation"]["params"]["n_iters"],
							clip=configuration["imputation"]["params"]["clip"]
						)
					elif imputer_name == 'iterative-sklearn':
						imp = IterativeImputer(
							max_iter=configuration["imputation"]["params"]["n_iters"],
							skip_complete=True, keep_empty_features=True
						)
					elif imputer_name == 'simple':
						imp = SimpleImputer(strategy='mean')
					else:
						raise ValueError("imputer name not supported")

					data_train_filled = imp.fit_transform(data_ms[:, :-1], data_ms[:, -1])
					data_filled = np.concatenate((data_train_filled, data_ms[:, -1].reshape(-1, 1)), axis=1)
					logger.debug(data_filled.shape)
				else:
					data_filled = data
					masks.append(np.isnan(data))

				datasets_filled.append(data_filled)

			# aggregation
			mask = np.concatenate(masks, axis=0)[:, :-1]
			logger.debug(mask.sum())
			train_set_filled = np.concatenate(datasets_filled, axis=0)
			X_train_filled, y_train_filled = train_set_filled[:, :-1], train_set_filled[:, -1]
			if ms_ratio == 0.0:
				X_train = X_train_filled.copy()
			# check two numerical array close
			logger.debug("{} {}".format(X_train_filled.shape, X_train.shape))

			# prediction and evaluation
			evaluator = Evaluator(
				task_type=data_config['task_type'],
				metrics=["model@acc", "imp@rmse", "imp@w2"],
				model=clf_name, tune_params=clf_tune_params,
				X_train=X_train, X_test=X_test, y_train=y_train_filled, y_test=y_test,
				mask=mask, seed=seed
			)

			eval_ret = evaluator.evaluation_imp(X_train_filled)
			eval_ret_all.append(eval_ret)

		# result cleaning
		results = self.experiment_result_processing(eval_ret_all)
		print(results)

		return results

	@staticmethod
	def experiment_pre_setup(configuration):
		pass

	@staticmethod
	def experiment_result_processing(results):
		metrics = list(results[0].keys())

		cleaned_ret = {}
		for metric in metrics:
			obj = {}
			for idx, item in enumerate(results):
				obj[idx] = item[metric]
			obj['avg'] = sum(obj.values()) / len(obj)
			cleaned_ret[metric] = obj

		return {
			"cleaned_result": cleaned_ret,
			"raw_result": results
		}
