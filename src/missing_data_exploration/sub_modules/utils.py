from typing import Dict, List
import matplotlib.pyplot as plt
from scipy import optimize
from typing import List
from scipy.special import expit
import numpy as np

def plot_lines(ret: Dict[str, List], emphasize_lines, name, range_=None, axes=None):
	if axes is None:
		fig, axes = plt.subplots(figsize=(10, 8))

	x = list(range(1, len(list(ret.values())[0]) + 1))
	for k, v in ret.items():
		if k in emphasize_lines:
			axes.plot(x, v, label=k, linewidth='3', alpha=1)
		else:
			axes.plot(x, v, label=k, linewidth='1', alpha=0.7)

	axes.set_xticks(x)
	axes.set_title(name)
	if range_:
		axes.set_ylim(*range_)

	if axes is None:
		#plt.savefig(name + '.png')
		plt.show()
	else:
		return axes


def pick_coeffs(
		X: np.ndarray,
		idxs_obs=None,
		idxs_nas=None,
		self_mask: bool = False,
) -> np.ndarray:
	if idxs_obs is None:
		idxs_obs = []
	if idxs_nas is None:
		idxs_nas = []

	n, d = X.shape
	if self_mask:
		coeffs = np.random.rand(d)
		Wx = X * coeffs
		coeffs /= np.std(Wx, 0)
	else:
		d_obs = len(idxs_obs)
		d_na = len(idxs_nas)
		coeffs = np.random.rand(d_obs, d_na)
		Wx = X[:, idxs_obs] @ coeffs
		coeffs /= np.std(Wx, 0, keepdims=True)
	return coeffs


def fit_intercepts(
		X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:

	if self_mask:
		d = len(coeffs)
		intercepts = np.zeros(d)
		for j in range(d):
			def f(x: np.ndarray) -> np.ndarray:
				return expit(X * coeffs[j] + x).mean().item() - p

			intercepts[j] = optimize.bisect(f, -50, 50)
	else:
		d_obs, d_na = coeffs.shape
		intercepts = np.zeros(d_na)
		for j in range(d_na):
			def f(x: np.ndarray) -> np.ndarray:
				return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p

			intercepts[j] = optimize.bisect(f, -50, 50)
	return intercepts


def show_missing_information(clients):
	import missingno
	import matplotlib.pyplot as plt
	for name, client in clients.items():
		df = client.X_train_ms_df
		fig, axes = plt.subplots(1, 2, figsize=(16, 12), squeeze=False)
		axes[0, 0].set_title("Missing proportion")
		missingno.bar(df, fontsize=8, color='lightblue', ax=axes[0, 0])
		axes[0, 1].set_title("Missing matrix")
		missingno.matrix(df, fontsize=8, ax=axes[0, 1], sparkline=False)
		plt.tight_layout()
		plt.title(name)
		plt.show()