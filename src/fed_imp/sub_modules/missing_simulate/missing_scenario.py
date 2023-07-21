import random
from copy import deepcopy

import numpy as np

MS_MECHANISM_MAPPING = {
	1: 'mcar',
	2: 'mar_quantile_left',
	3: 'mar_quantile_right',
	4: 'mar_quantile_mid',
	5: 'mar_quantile_tail',
	6: 'mar_sigmoid_left',
	7: 'mar_sigmoid_right',
	8: 'mar_sigmoid_mid',
	9: 'mar_sigmoid_tail',
	10: 'mary_left',
	11: 'mary_right',
	12: 'mary_mid',
	13: 'mary_tail',
	14: 'mnar_quantile_left',
	15: 'mnar_quantile_right',
	16: 'mnar_quantile_mid',
	17: 'mnar_quantile_tail',
	18: 'mary_sigmoid_left',
	19: 'mary_sigmoid_right',
	20: 'mnar_sigmoid_left',
	21: 'mnar_sigmoid_right',
	22: 'mnar_sigmoid_mid',
	23: 'mnar_sigmoid_tail',
}


########################################################################################################################
# Scenario functions
########################################################################################################################
def load_scenario2(
		n_clients, cols, mf_strategy, mm_strategy, mr_strategy, seed=0
):

	# missing features
	mf = missing_feature_strategy(n_clients, cols, mf_strategy, seed=seed)

	# missing mechanism
	mm, group_lens = ms_mechanism_strategy(n_clients, mm_strategy)

	# missing ratio
	mr = missing_ratio_strategy(mr_strategy, n_clients, group_lens, seed=seed)

	return [
		{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in zip(
			mr, mm, mf
		)]


########################################################################################################################
# Group splitting
########################################################################################################################
def group_splitting_unbalanced(original_array, group_lens):

	# missing ratios
	groups = []
	for group_len in group_lens:
		groups.append(original_array[:group_len])
		original_array = original_array[group_len:]
	return groups


########################################################################################################################
# Missing Feature Functions
########################################################################################################################
def missing_feature_strategy(n_clients, cols, strategy, seed=0):

	strategy, params = parse_strategy(strategy)

	# missing in all features
	if strategy == 'all':
		client_mf = [cols for _ in range(n_clients)]
	# missing in random features
	elif strategy == 'random':
		client_mf = []
		for _ in range(n_clients):
			np.random.seed(seed)
			feature_idx = np.random.choice(cols, size=int(len(cols) * 0.7), replace=False)
			client_mf.append(feature_idx)
	# missing in disjoint features
	elif strategy == 'disjoint':
		assert 'n' in params
		n_groups = int(params['n'])
		# split features into mutually exclusive groups
		features = np.array(cols)
		np.random.seed(seed)
		np.random.shuffle(features)
		feature_groups = np.array_split(features, n_groups)
		# each client randomly choose one group of feature sets
		client_mf = []
		for _ in range(n_clients):
			np.random.seed(seed)
			feature_cols = np.random.choice(feature_groups, size=1, replace=False)
			client_mf.append(feature_cols)
	else:
		raise ValueError('feature_split not found')

	return client_mf


########################################################################################################################
# Missing Ratio Strategy
########################################################################################################################
def missing_ratio_strategy(strategy, n_clients, group_lens=None, seed=0):

	strategy, params = parse_strategy(strategy)

	# missing ratio as random uniform
	if strategy == 'sequence':
		assert group_lens is not None
		group_mr = []
		for group_len in group_lens:
			np.random.seed(seed)
			# client_mr = [0.5 for _ in range(group_len)]
			client_mr = np.arange(0.3, 0.7, 0.4/group_len)
			group_mr.append(client_mr)
		client_mr = np.concatenate(group_mr)
	elif strategy == 'fixed':
		assert group_lens is not None
		ms_ratio = float(params['mr'])
		group_mr = []
		for group_len in group_lens:
			np.random.seed(seed)
			client_mr = np.ones(group_len) * ms_ratio
			group_mr.append(client_mr)
		client_mr = np.concatenate(group_mr)
	elif strategy == 'random':
		client_mr = np.random.uniform(low=0.3, high=0.7, size=n_clients)
	# missing ratio as random uniform in group
	elif strategy == 'random_in_group':
		assert group_lens is not None
		group_mr = []
		for group_len in group_lens:
			np.random.seed(seed)
			client_mr = np.random.uniform(low=0.3, high=0.7, size=group_len)
			group_mr.append(client_mr)
		client_mr = np.concatenate(group_mr)
	elif strategy == 'random_in_group2':
		assert group_lens is not None
		group_mr = []
		for group_len in group_lens:
			np.random.seed(seed)
			group_len1 = int(group_len * 0.5)
			group_len2 = group_len - group_len1
			client_mr1 = np.random.uniform(low=0.3, high=0.501, size=group_len1)
			client_mr2 = np.random.uniform(low=0.501, high=0.7, size=group_len2)
			client_mr = np.concatenate([client_mr1, client_mr2])
			group_mr.append(client_mr)
		client_mr = np.concatenate(group_mr)
	# missing ratio as clusters mixture of gaussian
	elif strategy == 'cluster':
		assert 'c' in params
		if params['c'] == '28':
			means = [0.2, 0.8]
			variances = [0.05, 0.05]
		elif params['c'] == '369':
			means = [0.3, 0.6, 0.9]
			variances = [0.05, 0.05, 0.05]
		elif params['c'] == '258':
			means = [0.2, 0.5, 0.8]
			variances = [0.05, 0.05, 0.05]
		else:
			raise ValueError('c not found')

		n_groups = len(means)

		# generate the size of each group by split the n_clients
		group_lens = np.array_split(np.arange(n_clients), n_groups)
		group_lens = [len(group) for group in group_lens]

		# generate missing ratio for each group
		group_mrs = []
		for idx, group_len in enumerate(group_lens):
			# clip missing ratio range between 0.05 to 0.95
			seed += 1
			np.random.seed(seed)
			group_mr = np.clip(
				np.random.normal(loc=means[idx], scale=variances[idx], size=group_len), 0.05,
				0.95
			)
			group_mrs.append(group_mr)

		# collect results of all groups to final result
		client_mr = np.concatenate(group_mrs)
	else:
		raise ValueError('strategy not found')

	return list(client_mr)


########################################################################################################################
# Missing Mechanism
########################################################################################################################
def ms_mechanism_strategy(n_clients, strategy):
	strategy, params = parse_strategy(strategy)

	if strategy == 'single':
		assert 'm' in params
		mech_cats = [int(params['m'])]
	elif strategy == 'mar_quantile_lr':
		mech_cats = [2, 3]
	elif strategy == 'mar_quantile_all':
		mech_cats = [2, 3, 4, 5]
	elif strategy == 'mar_sigmoid_lr':
		mech_cats = [6, 7]
	elif strategy == 'mar_sigmoid_all':
		mech_cats = [6, 7, 8, 9]
	elif strategy == 'mary_lr':
		mech_cats = [10, 11]
	elif strategy == 'mary_sigmoid_lr':
		mech_cats = [18, 19]
	elif strategy == 'mary_sigmoid_rl':
		mech_cats = [19, 18]
	elif strategy == 'mary_rl':
		mech_cats = [11, 10]
	elif strategy == 'mary_all':
		mech_cats = [10, 11, 12, 13]
	elif strategy == 'mnar_lr':
		mech_cats = [14, 15]
	elif strategy == 'mnar_rl':
		mech_cats = [15, 14]
	elif strategy == 'mnar_sigmoid_lr':
		mech_cats = [20, 21]
	elif strategy == 'mnar_sigmoid_rl':
		mech_cats = [21, 20]
	elif strategy == 'mnar_all':
		mech_cats = [14, 15, 16, 17]
	elif strategy == 'mnar_all':
		mech_cats = [20, 21, 22, 23]
	elif strategy == 'ignorable_ms_lr':
		mech_cats = [1, 2, 3]
	elif strategy == 'ignorable_ms_all':
		mech_cats = [1, 2, 3, 4, 5]
	elif strategy == 'nonignorable_ms_lr':
		mech_cats = [10, 11, 14, 15]
	elif strategy == 'nonignorable_ms_all':
		mech_cats = [10, 11, 12, 13, 14, 15, 16, 17]
	elif strategy == 'nonignorable_ms_random':
		mech_cats = random.sample([10, 11, 12, 13, 14, 15, 16, 17], 4)
	else:
		raise ValueError('strategy not found')

	# split clients based on number of different missing mechanisms
	split_func = params.get('sp', 'even')
	if split_func == 'even':
		n_groups = len(mech_cats)
		group_lens = np.array_split(np.arange(n_clients), n_groups)
		group_lens = [len(group) for group in group_lens]

		group_mechs = []
		for idx, group_len in enumerate(group_lens):
			mechanism = MS_MECHANISM_MAPPING[mech_cats[idx]]
			group_mechs.append([mechanism for _ in range(group_len)])

		client_mechs = np.concatenate(group_mechs)
	elif split_func == 'extreme':
		ratio = params.get('r', 0.05)
		ratio = float(ratio)
		if strategy not in [
			'mnar_lr', 'mary_lr', 'mary_rl', 'mnar_rl', 'mary_sigmoid_lr', 'mary_sigmoid_rl',
			'mnar_sigmoid_lr', 'mnar_sigmoid_rl'
		]:
			raise ValueError('extreme split only support mnar_lr and mary_lr')
		group_lens = [int(n_clients * ratio), int(n_clients * (1 - ratio))]
		group_mechs = []
		for idx, group_len in enumerate(group_lens):
			mechanism = MS_MECHANISM_MAPPING[mech_cats[idx]]
			group_mechs.append([mechanism for _ in range(group_len)])
		client_mechs = np.concatenate(group_mechs)
	elif split_func == 'extreme2':
		if strategy not in ['mnar_lr', 'mary_lr', 'mary_rl', 'mnar_rl', 'mary_sigmoid_lr', 'mary_sigmoid_rl']:
			raise ValueError('extreme split only support mnar_lr and mary_lr')
		group_lens = [int(n_clients * 0.05), int(n_clients * 0.95)]
		group_mechs = []
		for idx, group_len in enumerate(group_lens):
			mechanism = MS_MECHANISM_MAPPING[mech_cats[idx]]
			group_mechs.append([mechanism for _ in range(group_len)])
		client_mechs = np.concatenate(group_mechs)
		print(client_mechs)
	else:
		raise ValueError('split function not found')

	return list(client_mechs), group_lens


def parse_strategy(strategy):
	if '@' in strategy:
		strategy, params = strategy.split('@')
		param_dict = {}
		if params == '':
			return strategy, {}
		elif '_' in params:
			ret = params.split('_')
			for item in ret:
				if '=' in item:
					key, value = item.split('=')
					param_dict[key] = value
		elif '=' in params:
			key, value = params.split('=')
			param_dict[key] = value
		else:
			raise ValueError('invalid format of params')
		return strategy, param_dict
	else:
		return strategy, {}

########################################################################################################################
# Old Code
########################################################################################################################
def load_scenario1(name, n_clients, mechanism, cols, seed=0):

	# ==================================================================================================================
	# Cluster Missing Ratio
	# ==================================================================================================================
	if name == 'mr_two_cluster19_mf_all':
		ret = n_clusters_mr([0.1, 0.9], n_cluster=2, n_clients=n_clients, mechanism=mechanism, cols=cols)
	elif name == 'mr_two_cluster19_gaussian_mf_all':
		ret = n_clusters_mr(
			[0.1, 0.9], stds=[0.05, 0.05], n_cluster=2, n_clients=n_clients, mechanism=mechanism, cols=cols, seed=seed
		)
	elif name == 'mr_two_cluster19_unbalance_mf_all':
		ret = n_clusters_mr_unbalance([0.1, 0.9], [2, 8], n_clients=n_clients, mechanism=mechanism, cols=cols)
	elif name == 'mr_two_cluster19_mf_cross':
		ret = n_clusters_mr_mf([0.1, 0.9], n_cluster=2, n_clients=n_clients, mechanism=mechanism, cols=cols)
	elif name == 'mr_three_cluster369_mf_all':
		ret = n_clusters_mr([0.3, 0.6, 0.9], n_cluster=3, n_clients=n_clients, mechanism=mechanism, cols=cols)
	elif name == 'mr_three_cluster369_gaussian_mf_all':
		ret = n_clusters_mr(
			[0.3, 0.6, 0.9], stds=[0.05, 0.05, 0.05], n_cluster=3, n_clients=n_clients, mechanism=mechanism,
			cols=cols, seed=seed
		)
	elif name == 'mr_three_cluster369_unbalance_mf_all':
		ret = n_clusters_mr_unbalance([0.3, 0.6, 0.9], [2, 2, 6], n_clients=n_clients, mechanism=mechanism, cols=cols)
	elif name == 'mr_three_cluster369_mf_cross':
		ret = n_clusters_mr_mf([0.3, 0.6, 0.9], n_cluster=3, n_clients=n_clients, mechanism=mechanism, cols=cols)
	# ==================================================================================================================
	# Random Missing Ratio
	# ==================================================================================================================
	elif name == 'mr_random_mf_all':
		ret = random_uniform(n_clients=n_clients, mechanism=mechanism, cols=cols, feature_split='all', seed=seed)
	elif name == 'mr_random_mf_random':
		ret = random_uniform(n_clients=n_clients, mechanism=mechanism, cols=cols, feature_split='random', seed=seed)
	elif name == 'mr_random_mf_group2':
		ret = random_uniform(
			n_clients=n_clients, mechanism=mechanism, cols=cols, feature_split='group', k=2, seed=seed
		)
	elif name == 'mr_random_mf_group3':
		ret = random_uniform(
			n_clients=n_clients, mechanism=mechanism, cols=cols, feature_split='group', k=2, seed=seed
		)
	elif name == 'mr_mf_random':
		ret = random_uniform2(n_clients=n_clients, mechanism=mechanism, cols=cols, feature_split='all', seed=seed)
	# ==================================================================================================================
	# Different missing mechanisms
	# ==================================================================================================================
	elif name == "mr0.5_mm_mar_2cluster":
		ret = n_cluster_mm(n_cluster=2, mechanisms=[2, 3], ms_ratio=0.3, n_clients=n_clients, cols=cols, seed=0)
	elif name == "mr0.5_mm_mary_2cluster":
		ret = n_cluster_mm(n_cluster=2, mechanisms=[6, 7], ms_ratio=0.5, n_clients=n_clients, cols=cols, seed=0)
	elif name == "mr0.5_mm_mnar_2cluster":
		ret = n_cluster_mm(n_cluster=2, mechanisms=[12, 13], ms_ratio=0.5, n_clients=n_clients, cols=cols, seed=0)
	elif name == "mr_mm_mary_2cluster1":
		ret = n_cluster_mm(n_cluster=2, mechanisms=[6, 7], ms_ratio=[0.3, 0.7], n_clients=n_clients, cols=cols, seed=0)
	elif name == "mr_mm_mar_2cluster2":
		ret = n_cluster_mm2(n_cluster=2, mechanisms=[2, 3], ms_ratio=[], n_clients=n_clients, cols=cols, seed=0)
	elif name == "mr_mm_mary_2cluster2":
		ret = n_cluster_mm2(n_cluster=2, mechanisms=[6, 7], ms_ratio=[], n_clients=n_clients, cols=cols, seed=0)
	else:
		raise ValueError('Scenario name not found')

	return ret


def random_uniform(n_clients, mechanism, cols, feature_split, k=2, seed=0):

	# missing ratio
	np.random.seed(seed)
	client_mr = np.random.uniform(0, 1, n_clients)
	client_mr = np.clip(client_mr, 0.05, 0.95)

	# missing mechanism
	client_mm = [mechanism for _ in range(n_clients)]

	# missing features
	client_mf = missing_feature_strategy(n_clients, cols, feature_split, n_groups=k, seed=seed)

	return [
		{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in zip(
			client_mr, client_mm, client_mf
		)]


def random_uniform2(n_clients, mechanism, cols, feature_split, k=2, seed=0):

	# missing ratio
	np.random.seed(seed)
	client_mr = []
	for _ in range(n_clients):
		seed += 10390 % (2 ^ 32 - 1)
		np.random.seed(seed)
		ratios = np.random.uniform(0, 1, len(cols))
		ratios = np.clip(ratios, 0.05, 0.95)
		ratio_dict = {col: ratio for col, ratio in zip(cols, ratios)}
		client_mr.append(deepcopy(ratio_dict))

	# missing mechanism
	client_mm = [mechanism for _ in range(n_clients)]

	# missing features
	client_mf = missing_feature_strategy(n_clients, cols, feature_split, n_groups=k, seed=seed)

	return [
		{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in zip(
			client_mr, client_mm, client_mf
		)]


def n_clusters_mr(probs, stds=None, n_cluster=2, n_clients=10, mechanism='MCAR', cols=None, seed=0):

	assert len(probs) == n_cluster
	if stds is not None:
		assert len(stds) == n_cluster

	# missing ratios
	clients = np.ones(n_clients)  # client ids
	groups = np.array_split(clients, n_cluster)  # split clients into 2 groups
	if stds is not None:
		np.random.seed(seed)
		for i in range(n_cluster):
			prob_array = np.random.normal(probs[i], stds[i], len(groups[i]))
			prob_array = np.clip(prob_array, 0.05, 0.95)
			groups[i] = prob_array
	else:
		for i in range(n_cluster):
			groups[i] = groups[i] * probs[i]
	client_mr = np.concatenate(groups)

	# missing mechanism
	client_mm = [mechanism for _ in range(n_clients)]

	# missing features
	client_mf = [cols for _ in range(n_clients)]

	return [{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in
	        zip(client_mr, client_mm, client_mf)]


def n_clusters_mr_unbalance(probs, group_lens, n_clients=10, mechanism='MCAR', cols=None):

	assert len(probs) == len(group_lens)
	assert sum(group_lens) == n_clients

	# missing ratios
	clients = np.ones(n_clients)  # client ids
	groups = group_splitting_unbalanced(clients, group_lens)
	for i in range(len(group_lens)):
		groups[i] = groups[i] * probs[i]
	client_mr = np.concatenate(groups)

	# missing mechanism
	client_mm = [mechanism for _ in range(n_clients)]

	# missing features
	client_mf = [cols for _ in range(n_clients)]

	return [{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in
	        zip(client_mr, client_mm, client_mf)]


def n_clusters_mr_mf(probs, n_cluster=2, n_clients=10, mechanism='MCAR', cols=None, seed=201030):

	assert len(probs) == n_cluster

	# missing ratios
	clients = np.ones(n_clients)  # client ids
	groups = np.array_split(clients, n_cluster)  # split clients into 2 groups
	for i in range(n_cluster):
		groups[i] = groups[i] * probs[i]
	client_mr = np.concatenate(groups)

	# missing mechanism
	client_mm = [mechanism for _ in range(n_clients)]

	# missing features
	client_mf = []
	features = np.array(cols)
	np.random.seed(seed)
	np.random.shuffle(features)
	feature_groups = np.array_split(features, n_cluster)

	for client_group, feature_group in zip(groups, feature_groups):
		# get cols for each group
		cols = list(feature_group)
		for _ in range(len(client_group)):
			client_mf.append(cols)

	return [{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in
	        zip(client_mr, client_mm, client_mf)]


def n_cluster_mm(n_cluster, mechanisms, ms_ratio, n_clients=10, cols=None, seed=0):

	# missing ratios
	if isinstance(ms_ratio, float):
		client_mr = [ms_ratio for _ in range(n_clients)]
	elif isinstance(ms_ratio, list):
		clients = np.ones(n_clients)  # client ids
		groups = np.array_split(clients, n_cluster)  # split clients into 2 groups
		for i in range(n_cluster):
			groups[i] = groups[i] * ms_ratio[i]
		client_mr = np.concatenate(groups)
		client_mr = list(client_mr)
	else:
		raise NotImplementedError

	# missing mechanism
	clients = np.ones(n_clients)  # client ids
	groups = np.array_split(clients, n_cluster)  # split clients into 2 groups
	for i in range(n_cluster):
		groups[i] = groups[i] * mechanisms[i]
	client_mm_ids = np.concatenate(groups)

	client_mm = [MS_MECHANISM_MAPPING[mechanism_id] for mechanism_id in client_mm_ids]

	# missing features
	client_mf = [cols for _ in range(n_clients)]

	return [{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in
	        zip(client_mr, client_mm, client_mf)]


def n_cluster_mm2(n_cluster, mechanisms, ms_ratio, n_clients=10, cols=None, seed=0):

	# missing ratios
	if isinstance(ms_ratio, float):
		client_mr = [ms_ratio for _ in range(n_clients)]
	elif isinstance(ms_ratio, list):
		ms_ratio = [0.3, 0.4, 0.5, 0.6, 0.7]
		groups = [np.array(ms_ratio), np.array(ms_ratio)]
		client_mr = np.concatenate(groups)
		client_mr = list(client_mr)
	else:
		raise NotImplementedError

	# missing mechanism
	clients = np.ones(n_clients)  # client ids
	groups = np.array_split(clients, n_cluster)  # split clients into 2 groups
	for i in range(n_cluster):
		groups[i] = groups[i] * mechanisms[i]
	client_mm_ids = np.concatenate(groups)

	client_mm = [MS_MECHANISM_MAPPING[mechanism_id] for mechanism_id in client_mm_ids]

	# missing features
	client_mf = [cols for _ in range(n_clients)]

	return [{"missing_ratio": mr, "missing_mechanism": mm, "missing_features": mf} for mr, mm, mf in
	        zip(client_mr, client_mm, client_mf)]
