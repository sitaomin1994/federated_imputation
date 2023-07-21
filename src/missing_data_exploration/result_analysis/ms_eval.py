def clean_result_ms_eval(ret_dict):

	cleaned_ret = dict()
	# add configuration info
	config = ret_dict['params']['config']
	cleaned_ret.update({
		'dataset_name': config['dataset']['name'],
		'ms_ratio': config['missing_simulate']['ms_ratio'],
		'ms_feature_ratio': config['missing_simulate']['ms_features'],
		"ms_mechanism": config['missing_simulate']['ms_mechanism'],
		"imputation": config['imputation']['name'],
		"classifier": config['classifier']['name'],
		"n_rounds": config['fed_imp']['n_rounds'],
	})

	# results
	results = ret_dict['results']["cleaned_result"]
	cleaned_ret.update({
		"acc": results['accuracy']["avg"],
		'roc_auc': results['roc_auc']["avg"],
		"rmse": results['imp@rmse']["avg"],
		"ws": results['imp@w2']["avg"],
	})

	return cleaned_ret


def clean_ret_ms_eval_all(ret_dict):

	cleaned_ret = dict()
	# add configuration info
	config = ret_dict['params']['config']
	cleaned_ret.update({
		'dataset_name': config['dataset']['name'],
		'ms_ratio': config['missing_simulate']['ms_ratio'],
		'ms_feature_ratio': config['missing_simulate']['ms_features'],
		"ms_mechanism": config['missing_simulate']['ms_mechanism'],
		"imputation": config['imputation']['name'],
		"classifier": config['clf_model']['name'],
		"n_rounds": config['fed_imp']['n_rounds'],
	})

	# results
	results = ret_dict['results']["cleaned_result"]
	cleaned_ret.update(
		{
			"ret": results
		}
	)

	return cleaned_ret
