import numpy as np
from .ms_simulate import mcar_simulate, mar_simulate, mnar_simulate
from .missing_scenario import load_scenario1, load_scenario2


def add_missing(train_data_list, scenario, cols, seed=201030):
	mr_strategy = scenario['mr_strategy']
	mf_strategy = scenario['mf_strategy']
	mm_strategy = scenario['mm_strategy']
	ret = load_scenario2(
		n_clients=len(train_data_list), cols=cols, mr_strategy=mr_strategy, mf_strategy=mf_strategy,
		mm_strategy=mm_strategy, seed=seed
	)
	train_ms_list = []
	for i in range(len(train_data_list)):
		data = train_data_list[i]
		X_train, y_train = data[:, :-1], data[:, -1]
		missing_ratio = ret[i]['missing_ratio']
		missing_mechanism = ret[i]['missing_mechanism']
		missing_features = ret[i]['missing_features']
		seed = (seed + i * 10089) % (2 ^ 32 - 1)
		X_train_ms = simulate_nan_new(X_train, y_train, missing_features, missing_ratio, missing_mechanism, seed)
		train_ms_list.append(np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1).copy())

	return train_ms_list


########################################################################################################################
# Simulate missing for one client
########################################################################################################################
def simulate_nan_new(
		X_train, y_train, cols, missing_ratio, mechanism='mcar', seed=201030
):

	if mechanism == 'mcar':
		X_train_ms = mcar_simulate.simulate_nan_mcar(X_train, cols, missing_ratio, seed)
	elif mechanism == 'mar_quantile_left':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_quantile(
			X_train, cols, missing_ratio, missing_func='left', obs=True, seed=seed
		)
	elif mechanism == 'mar_quantile_right':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_quantile(
			X_train, cols, missing_ratio, missing_func='right', obs=True, seed=seed
		)
	elif mechanism == 'mar_quantile_mid':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_quantile(
			X_train, cols, missing_ratio, missing_func='mid', obs=True, seed=seed
		)
	elif mechanism == 'mar_quantile_tail':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_quantile(
			X_train, cols, missing_ratio, missing_func='tail', obs=True, seed=seed
		)
	elif mechanism == 'mar_sigmoid_left':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
			X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
		)
	elif mechanism == 'mar_sigmoid_right':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
			X_train, cols, missing_ratio, missing_func='right', obs=True, k='all', seed=seed
		)
	elif mechanism == 'mar_sigmoid_mid':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1]-1)
		X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
			X_train, cols, missing_ratio, missing_func='mid', obs=True, k='all', seed=seed
		)
	elif mechanism == 'mar_sigmoid_tail':
		if len(cols) == X_train.shape[1]:
			cols = np.arange(0, X_train.shape[1] - 1)
		X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
			X_train, cols, missing_ratio, missing_func='tail', obs=True, k='all', seed=seed
		)
	elif mechanism == 'mary_left':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='left', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mary_right':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='right', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mary_sigmoid_left':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='left', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mary_sigmoid_right':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func='right', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mary_mid':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='mid', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mary_tail':
		data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)
		data_ms = mar_simulate.simulate_nan_mary_quantile(data, cols, missing_ratio, missing_func='tail', seed=seed)
		X_train_ms = data_ms[:, :-1]
	elif mechanism == 'mnar_sigmoid_left':
		data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(X_train, cols, missing_ratio, missing_func='left', seed=seed)
		X_train_ms = data_ms
	elif mechanism == 'mnar_sigmoid_right':
		data_ms = mnar_simulate.simulate_nan_mnar_sigmoid(X_train, cols, missing_ratio, missing_func='right',
		                                                  seed=seed)
		X_train_ms = data_ms
	elif mechanism == 'mnar_quantile_left':
		data_ms = mnar_simulate.simulate_nan_mnar_quantile(
			X_train, cols, missing_ratio, missing_func='left', seed=seed)
		X_train_ms = data_ms
	elif mechanism == 'mnar_quantile_right':
		data_ms = mnar_simulate.simulate_nan_mnar_quantile(
			X_train, cols, missing_ratio, missing_func='right', seed=seed)
		X_train_ms = data_ms
	elif mechanism == 'mnar_quantile_mid':
		data_ms = mnar_simulate.simulate_nan_mnar_quantile(X_train, cols, missing_ratio, missing_func='mid',
		                                                   seed=seed)
		X_train_ms = data_ms
	elif mechanism == 'mnar_quantile_tail':
		data_ms = mnar_simulate.simulate_nan_mnar_quantile(
			X_train, cols, missing_ratio, missing_func='tail', seed=seed)
		X_train_ms = data_ms
	else:
		raise NotImplementedError

	return X_train_ms
