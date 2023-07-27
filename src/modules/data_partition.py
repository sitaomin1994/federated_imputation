from typing import List

import pandas as pd

from .sampling import dirichlet_noniid_partition, generate_alphas
from sklearn.model_selection import train_test_split
import numpy as np


def data_partition(strategy, params, data, n_clients, seed=201030, regression=False) -> List[np.ndarray]:
	strategy, params = strategy.split('@')[0], dict([param.split('=') for param in strategy.split('@')[1:]])
	print(strategy, params)
	if strategy == 'full':
		return [data.copy() for _ in range(n_clients)]
	elif strategy == 'sample-evenly':
		sample_fracs = [1 / n_clients for _ in range(n_clients)]
		ret = []
		for idx, sample_frac in enumerate(sample_fracs):
			new_seed = seed + idx * seed + 990983
			# new_seed = seed
			if regression:
				_, X_test, _, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32)
				)
			else:
				_, X_test, _, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
				)
			ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
		return ret
	elif strategy == 'sample':
		sample_frac = int(params['p']) / data.shape[0]
		print(sample_frac, data.shape[0])
		ret = []
		sample_fracs = [sample_frac for _ in range(n_clients)]
		# print(pd.DataFrame(data[:, -1]).value_counts())
		for idx, sample_frac in enumerate(sample_fracs):
			new_seed = seed + idx * seed + 990983
			# new_seed = seed
			if regression:
				X_train, X_test, y_train, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32)
				)
			else:
				X_train, X_test, y_train, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
				)
			ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
		return ret
	elif strategy == '2case':
		sample_frac1 = int(params['s1']) / data.shape[0]
		sample_frac2 = int(params['s2']) / data.shape[0]
		sample_fracs = [sample_frac1 for _ in range(int(n_clients*0.5))] + [sample_frac2 for _ in range(int(n_clients*0.5))] 
		ret = []
		for idx, sample_frac in enumerate(sample_fracs):
			new_seed = seed + idx * seed + 990983
			# new_seed = seed
			if regression:
				X_train, X_test, y_train, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32)
				)
			else:
				X_train, X_test, y_train, y_test = train_test_split(
					data[:, :-1], data[:, -1], test_size=sample_frac,
					random_state=(new_seed) % (2 ** 32), stratify=data[:, -1]
				)
			ret.append(np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1).copy())
		return ret
	elif strategy == 'dirichlet':
		alpha = params.get('alpha', 1)
		alphas = generate_alphas(alpha, n_clients)
		indices = dirichlet_noniid_partition(alphas, data)
		return [data.loc[indices[i], :].copy() for i in range(n_clients)]
	else:
		raise ValueError('partition strategy not found')
