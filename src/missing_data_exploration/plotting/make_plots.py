import os
from datetime import datetime
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from config import ROOT_DIR, settings
from .plot_lines import plot_lines, plot_lines_bar
from src.experiment_management.result_management import collect_all_results
from ..result_analysis import ms_eval


def make_plots_factory(exp_dir, levels_dir, plot_params):

	# collect all results
	rets = collect_all_results(exp_dir, levels_dir, clean_func=ms_eval.clean_ret_ms_eval_all)

	# filter needed information for plotting
	filter_params = plot_params['filter']
	rets = filter_ret(rets, filter_params)

	# plot based on plot type
	plot_type = plot_params['type']
	base = os.path.join(ROOT_DIR, settings['experiment_result_dir'])
	output_dir = os.path.join(base, settings['processed_result_dir'], exp_dir.replace("/", "\\"))
	name = "_".join(levels_dir)
	if plot_type == 'line_imp_vs_ms_ratio':
		filer_name = "_".join(["_".join(v) for k, v in filter_params.items() if len(v) > 0])
		output_filename = os.path.join(
			output_dir, "plot_{}_{}.png".format(name, filer_name)
			)
		print(output_filename)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		make_plot_line_bar_ms_ratio(rets, plot_params['field'], metrics=None, output_filename=output_filename)


def filter_ret(rets, filter_params: Dict[str, list]):

	filtered_rets = []
	for ret in rets:
		# filter based on all parameters
		valid = True
		for key, values in filter_params.items():
			if len(values) > 0:
				if ret[key] not in values:
					valid = False
					break
		if valid:
			filtered_rets.append(ret)

	return filtered_rets


########################################################################################################################
# Plotting functions
########################################################################################################################
def make_plot_line_bar_ms_ratio(rets, field, metrics=None, output_filename=None):
	if metrics is None:
		metrics = ['accuracy', 'roc_auc', 'imp@rmse', 'imp@w2']

	# expand rets to long format
	long_rets = []
	for item in rets:
		base_dict = {k: v for k, v in item.items() if k != 'ret'}
		all_rounds_metrics = {}
		for metric, value in item['ret'].items():
			for round_key, metric_value in value.items():
				if round_key != 'avg':
					if round_key not in all_rounds_metrics:
						all_rounds_metrics[round_key] = {metric: metric_value}
					else:
						all_rounds_metrics[round_key][metric] = metric_value
		for key in all_rounds_metrics:
			long_rets.append({**base_dict, **all_rounds_metrics[key], 'round': key})

	# convert to dataframe
	df = pd.DataFrame(long_rets)

	n_cols = 4 if len(metrics) > 4 else len(metrics)
	n_rows = (len(metrics) - 1) // 4 + 1

	fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(n_cols * 5, n_rows * 5))
	for metric_idx, metric in enumerate(metrics):
		plot_lines_bar(df, metric, field, axes=axes[metric_idx // 4, metric_idx % 4])

	plt.tight_layout()
	if output_filename:
		fig.savefig(output_filename, format='png')
	#fig.savefig(file_path, format='png')
	plt.show()
	plt.close(fig)
