from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_lines(ret: Dict[str, List], emphasize_lines, name, x, range_=None, axes=None):

	if axes is None:
		fig, axes = plt.subplots(figsize=(10, 8))

	for k, v in ret.items():
		if k in emphasize_lines:
			axes.plot(x, v, label=k, linewidth='3', alpha=1, marker='o', markersize=8)
		else:
			axes.plot(x, v, label=k, linewidth='1', alpha=0.7, marker='o', markersize=5)

	axes.set_xticks(x)
	axes.set_xlabel('Missing Ratio')
	axes.set_title(name)
	if range_:
		axes.set_ylim(*range_)

	if axes is None:
		# plt.savefig(name + '.png')
		plt.show()
	else:
		return axes


def plot_lines_bar(ret: pd.DataFrame, metric, field, axes=None):

	if axes is None:
		fig, axes = plt.subplots(figsize=(10, 8))

	err_func = lambda x: (np.min(x), np.max(x))

	sns.lineplot(ret, x="ms_ratio", y=metric, hue=field, sort=False, err_style='band',
	             errorbar=err_func, markers=True, ax=axes)
	axes.set_xlabel('Missing Ratio')
	axes.set_xticks(ret['ms_ratio'].unique())
	axes.set_title(metric)

	if axes is None:
		# plt.savefig(name + '.png')
		plt.show()
	else:
		return axes
