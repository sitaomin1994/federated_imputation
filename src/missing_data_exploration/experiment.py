import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer

from src.modules.data_preprocessing import load_data
from .sub_modules.data_spliting import split_train_test
from src.missing_data_exploration.sub_modules.ms_simulate.ms_simulate_hy import simulate_nan
from src.modules.iterative_imputation.local_imputer import LocalIterativeImputer
from src.modules.evaluation.evaluation import Evaluator
from loguru import logger
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor


class ExperimentMSEval:

	name = 'imp_eval'

	def __init__(self, debug=False):
		self.debug = debug

	def run_experiment(self, configuration):
		# missing params
		ms_ratio = configuration['missing_simulate']['ms_ratio']
		ms_features = configuration['missing_simulate']['ms_features']
		ms_mechanism = configuration['missing_simulate']['ms_mechanism']

		# classification model
		clf_name = configuration['clf_model']['name']
		clf_tune_params = configuration['clf_model']['tune_params']

		# imputer
		imputer_name = configuration['imputation']['name']

		# load dataset
		dataset_name = configuration['dataset']['name']
		data, data_config = load_data(dataset_name)

		# n_rounds splitting
		n_splits = configuration['fed_imp']['n_rounds']
		seed = configuration['fed_imp']['seed']
		train_test_sets = split_train_test(data, n_splits, seed)

		# missing simulate
		eval_ret_all = []
		for idx, (train_set, test_set) in enumerate(train_test_sets):

			# pre setup
			X_train, y_train = train_set.values[:, :-1], train_set.values[:, -1]
			X_test, y_test = test_set.values[:, :-1], test_set.values[:, -1]
			logger.info("Round - {}".format(idx))

			# missing simulate
			if ms_ratio != 0:
				ret = simulate_nan(
					train_set.values, ms_ratio, ms_mechanism, p_na_cols=ms_features,
					seed=(seed + idx * 1000809) % 1000009890977
				)

				train_ms = ret['train_data_ms']

				# imputation
				if imputer_name == 'iterative-local':
					imp = LocalIterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						clip=configuration["imputation"]["params"]["clip"]
					)
				elif imputer_name == 'iterative-sklearn':
					imp = IterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						skip_complete=True,
						random_state=seed
						# keep_empty_features=True
					)
				elif imputer_name == 'iterative-sklearn-ds':
					imp = IterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						skip_complete=True, imputation_order='random',
						random_state=seed
						# keep_empty_features=True
					)
				elif imputer_name == 'iterative-sklearn-clip':
					min_array = np.nanmin(train_ms[:, :-1], axis=0)
					max_array = np.nanmax(train_ms[:, :-1], axis=0)
					imp = IterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						skip_complete=True, min_value=min_array, max_value=max_array,
						random_state=seed
						# keep_empty_features=True
					)
				elif imputer_name == 'iterative-sklearn-ridgecv':
					min_array = np.nanmin(train_ms[:, :-1], axis=0)
					max_array = np.nanmax(train_ms[:, :-1], axis=0)
					imp = IterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						skip_complete=True, min_value=min_array, max_value=max_array,
						random_state=seed, estimator=RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
					)
				elif imputer_name == 'iterative-sklearn-rf':
					min_array = np.nanmin(train_ms[:, :-1], axis=0)
					max_array = np.nanmax(train_ms[:, :-1], axis=0)
					imp = IterativeImputer(
						max_iter=configuration["imputation"]["params"]["n_iters"],
						skip_complete=True, min_value=min_array, max_value=max_array,
						random_state=seed, estimator=RandomForestRegressor(
							n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
						)
					)
				elif imputer_name == 'knn':
					imp = KNNImputer(
						n_neighbors=configuration["imputation"]["params"]["n_neighbors"],
					)
				elif imputer_name == 'simple':
					imp = SimpleImputer(strategy='mean')
				else:
					raise ValueError("imputer name not supported")

				X_train_ms, y_train = train_ms[:, :-1], train_ms[:, -1]
				X_train_filled = imp.fit_transform(X_train_ms, y_train)
			else:
				X_train_ms = X_train
				X_train_filled = X_train

			# prediction and evaluation
			mask = np.isnan(X_train_ms)
			evaluator = Evaluator(
				task_type=data_config['task_type'],
				metrics=["model@acc", "imp@rmse", "imp@w2"],
				model=clf_name, tune_params=clf_tune_params,
				X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
				mask=mask, seed=seed
			)

			eval_ret = evaluator.evaluation_imp(X_train_filled)
			eval_ret_all.append(eval_ret)

		# result cleaning
		results = self.experiment_result_processing(eval_ret_all)
		print(
			"accu {} roc_auc {} rmse {} ".format(
				results['cleaned_result']['accuracy']['avg'],
				results['cleaned_result']['roc_auc']['avg'],
				results['cleaned_result']['imp@rmse']['avg']
			)
		)
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
