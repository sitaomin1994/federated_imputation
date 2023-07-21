
def clean_result_ms_eval_partition(ret_dict):

	cleaned_ret = dict()
	# add configuration info
	config = ret_dict['params']['config']
	cleaned_ret.update({
		'dataset_name': config['dataset']['name'],
		'ms_ratio': config['missing_simulate']['ms_ratio'],
		'ms_feature_ratio': config['missing_simulate']['ms_features'],
		"data_partition": config['data_partition']['k'],
		"ms_mechnism": config['missing_simulate']['ms_mechanism'],
		"imputation": config['imputation']['name'],
		"n_rounds": config['fed_imp']['n_rounds'],
	})

	# results
	results = ret_dict['results']["cleaned_result"]
	cleaned_ret.update({
		"acc": results['accuracy']["avg"],
		'roc': results['roc_auc']["avg"],
		"rmse": results['imp@rmse']["avg"],
		"ws": results['imp@w2']["avg"],
	})

	return cleaned_ret