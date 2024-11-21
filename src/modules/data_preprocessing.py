import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer
#from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .data_prep_utils import (
	normalization, move_target_to_end, convert_gaussian, drop_unique_cols, one_hot_categorical,
)
from .data_prep_his import (
	process_NHIS_income, process_heart, process_codrna, process_skin, process_codon, process_sepsis,
	process_diabetic, process_diabetic2, process_cardio, process_mimiciii_mortality, process_genetic,
	process_mimiciii_mo2, process_mimic_icd, process_mimic_icd2, process_mimic_mo, process_mimic_los
)
import json


########################################################################################################################
# Iris
########################################################################################################################
def process_iris(normalize=True, verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.3
	data = pd.read_csv("./data/iris/iris.csv", header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	data['5'] = data['5'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
	target_col = '5'

	if normalize:
		data = normalization(data, target_col)

	# move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	# correlation
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		'num_cols': data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Breast
########################################################################################################################
def process_breast(normalize=True, verbose=False, threshold=0.3):
	if threshold is None:
		threshold = 0.3
	data = pd.read_csv("./data/breast/data.csv", header=0)
	data = data.drop(["id"], axis=1)
	data = data.dropna()
	data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

	target_col = 'diagnosis'

	if normalize:
		data = normalization(data, target_col)

	# drop co-linear features
	drop_list = ['perimeter_mean', 'radius_mean', 'compactness_mean', 'concave points_mean', 'radius_se',
				 'perimeter_se', 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst',
				 'compactness_se', 'concave points_se', 'texture_worst', 'area_worst']
	data = data.drop(drop_list, axis=1)  # do not modify x, we will use it later

	# move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	# correlation
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': 'diagnosis',
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != 'diagnosis'],
		'num_cols': data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine white
########################################################################################################################
def process_white(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
	data = data.dropna()
	# data['target'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1, axis=1)
	# data = data.drop(['quality'], axis=1)

	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine red
########################################################################################################################
def process_red(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
	data = data.dropna()

	# data['target'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1, axis=1)
	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Wine all three
########################################################################################################################
def process_wine_three(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	# merge data
	data_white = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
	data_white['type'] = 0
	data_red = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
	data_red['type'] = 1
	data = pd.concat([data_white, data_red], axis=0)
	data = data.dropna()

	# label
	data['quality'] = data.apply(lambda row: 0 if row['quality'] <= 5 else 1 if row['quality'] <= 7 else 2, axis=1)
	print(data['quality'].value_counts())
	target_col = 'quality'
	data = move_target_to_end(data, target_col)

	# normalize
	if normalize:
		data = normalization(data, target_col, categorical_cols=['type'])

	# correlation
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1] - 1)],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Spambase
########################################################################################################################
def process_spam(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/spambase/spambase.csv", header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '58'

	# Normalization
	if normalize:
		data = normalization(data, target_col)

	# move target to the end of the dataframe
	data = move_target_to_end(data, target_col)

	# split train and test
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	print(important_features)

	data_config = {
		'target': '58',
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Blocks
########################################################################################################################
def process_blocks(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/block/page-blocks.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '11'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	correlation_ret = data.corrwith(data['11'], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Ecoli
########################################################################################################################
def process_ecoli(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/ecoli/ecoli.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data = data.drop([0], axis=1)
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '8'
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = data[target_col].map(
		{'cp': 0, 'im': 1, 'pp': 2, 'imU': 3, 'om': 4, 'omL': 5, 'imL': 5, 'imS': 5}
	)
	data = data.dropna()
	data = move_target_to_end(data, target_col)
	# data[target_col] = pd.factorize(data[target_col])[0]
	data = drop_unique_cols(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Glass
########################################################################################################################
def process_glass(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/glass/glass.csv", delimiter=',', header=None)
	data = data.dropna()
	data = data.drop([0], axis=1)
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '10'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data = data[data[target_col] != 6]
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Optical Digits
########################################################################################################################
def process_optdigits(normalize=True, verbose=False, threshold=0.5):
	if threshold is None:
		threshold = 0.5
	data_test = pd.read_csv("./data/optdigits/optdigits_test.csv", delimiter=',', header=None)
	data_train = pd.read_csv("./data/optdigits/optdigits_train.csv", delimiter=',', header=None)
	data = pd.concat([data_test, data_train]).reset_index(drop=True)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '65'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Segmentation
########################################################################################################################
def process_segmentation(normalize=True, verbose=False, threshold=0.5):
	if threshold is None:
		threshold = 0.5
	data_test = pd.read_csv("./data/segment/segmentation_test.csv", delimiter=',')
	data_train = pd.read_csv("./data/segment/segmentation.csv", delimiter=',')
	data = pd.concat([data_test, data_train]).reset_index(drop=True)
	data = data.dropna()
	target_col = 'TARGET'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col].value_counts()
	data[target_col] = data[target_col].map(
		{
			'BRICKFACE': 0,
			'SKY': 1,
			'FOLIAGE': 2,
			'CEMENT': 3,
			'WINDOW': 4,
			'PATH': 5,
			'GRASS': 6,
		}
	)
	data = data.drop(["REGION-PIXEL-COUNT"], axis=1)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Sonar
########################################################################################################################
def process_sonar(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/sonar/sonar.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '61'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = data[target_col].map(
		{
			'R': 0,
			'M': 1,
		}
	)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Sensor
########################################################################################################################
def process_sensor(normalize=True, verbose=False, threshold=0.2, pca=False):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/sensor/Sensorless_drive_diagnosis.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '49'

	if pca:
		pca = PCA(n_components=10)
		pca.fit(data.drop(target_col, axis=1))
		data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data = data[important_features + [target_col]]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Waveform
########################################################################################################################
def process_waveform(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/waveform/waveform-5000.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '41'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col] = data[target_col].map({'N': 0, 'P': 1})
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Yeast
########################################################################################################################
def process_yeast(normalize=True, verbose=False, threshold=0.4):
	if threshold is None:
		threshold = 0.4
	data = pd.read_csv("./data/yeast/yeast.csv", delimiter='\s+', header=None)
	data = data.drop([0], axis=1)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '9'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)
	data[target_col].value_counts()
	data[target_col] = data[target_col].map(
		{
			'CYT': 0,
			'NUC': 1,
			'MIT': 2,
			'END': 3,
			'ME3': 4,
			'ME2': 5,
			'ME1': 6,
			'EXC': 7,
			'VAC': 8,
			'POX': 9,
			'ERL': 10,
		}
	)
	data = data[data[target_col] != 10]
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Letter
########################################################################################################################
def process_letter(normalize=True, verbose=False, threshold=0.3):
	if threshold is None:
		threshold = 0.3
	data = pd.read_csv("./data/letter/letter-recognition.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'
	data = move_target_to_end(data, target_col)
	# convert columns to gaussian
	data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)
	# data[target_col].value_counts()
	data[target_col] = data[target_col].map(lambda x: ord(x) - ord('A'))
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# Raisin
########################################################################################################################
def process_raisin(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2
	data = pd.read_csv("./data/raisin/Raisin_Dataset.csv", delimiter=',')
	data = data.dropna()
	target_col = 'Class'
	data = move_target_to_end(data, target_col)
	if normalize:
		data = normalization(data, target_col)
	data[target_col] = data[target_col].map({"Kecimen": 0, "Besni": 1})
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Dermatology
########################################################################################################################
def process_dermatology(normalize=False, verbose=False, threshold=0.5):
	if threshold is None:
		threshold = 0.5
	data = pd.read_csv("./data/dermatology/dermatology.csv", delimiter=',', na_values='?', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '35'
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	if normalize:
		data = normalization(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Pima diabetes
########################################################################################################################
def process_pima_diabetes(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/pima_indianas_diabetes/diabetes.csv", delimiter=',', header=0)
	data = data.dropna()
	data['Outcome'].value_counts()
	target_col = 'Outcome'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	# print(correlation_ret)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Telugu Vowel
########################################################################################################################
def process_telugu_vowel(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/telugu_vowel/telugu.csv", delimiter=',', header=0)
	data = data.dropna()
	target_col = 'class'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	important_features = data.columns.tolist()
	important_features.remove(target_col)

	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'image'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Telugu Tabular
########################################################################################################################
def process_telugu_tabular(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/telugu_tabular/telugu.csv", delimiter='\s+', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	important_features = data.columns.tolist()
	important_features.remove(target_col)

	data[target_col] = pd.factorize(data[target_col])[0]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Wine
########################################################################################################################
def process_wine(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/wine2/wine.csv", delimiter=',', header=None)
	data = data.dropna()
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = '1'

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_wifi(normalize=True, verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.1

	data = pd.read_csv("./data/wifi_localization/wifi_localization.csv", delimiter='\s+', header=None)
	data = data.dropna()
	# data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	target_col = 7

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_adult(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):

	if threshold is None:
		threshold = 0.1
	# retain_threshold = 0.05

	columns = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
			   "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
			   "Hours per week", "Country", "Target"]
	types = {
		0: int, 1: str, 2: int, 3: str, 4: int, 5: str, 6: str, 7: str, 8: str, 9: str, 10: int,
		11: int, 12: int, 13: str, 14: str
	}

	data_train = pd.read_csv(
		"./data/adult/adult_train.csv", names=columns, na_values=['?'], sep=r'\s*,\s*', engine='python',
		dtype=types
	)
	data_test = pd.read_csv(
		"./data/adult/adult_test.csv", names=columns, na_values=['?'], sep=r'\s*,\s*', engine='python',
		dtype=types
	)
	data = pd.concat([data_train, data_test], axis=0)
	data = data.dropna()
	col_drop = ["Country", "Education", "fnlwgt"]
	data = data.drop(col_drop, axis=1)
	# target
	target_col = 'Target'
	data[target_col] = data[target_col].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

	# convert categorical to numerical
	if pca:
		cat_cols = ["Sex", "Martial Status", "Relationship", "Race", "Workclass", "Occupation"]
		data_oh = pd.get_dummies(data.drop(target_col, axis=1), columns=cat_cols, drop_first=True)
		data = pd.concat([data[target_col], data_oh], axis=1)
		data.reset_index(drop=True, inplace=True)
		print(data.shape)

		pca = PCA(n_components=20)
		pca.fit(data.drop(target_col, axis=1))
		data = pd.concat([data[target_col], pd.DataFrame(pca.transform(data.drop(target_col, axis=1)))], axis=1)
	else:
		for col in ["Sex", "Martial Status", "Relationship", "Race", "Workclass", "Occupation"]:
			values = data[col].value_counts().index.tolist()
			corr_y = []
			for value in values:
				corr_y_data = data[data[col] == value][target_col].value_counts(normalize=True)
				corr = corr_y_data[0] / corr_y_data[1]
				corr_y.append(corr)

			sorted_values = sorted(values, key=lambda x: corr_y[values.index(x)])
			np.random.seed(31)
			np.random.shuffle(sorted_values)
			mapping = {value: idx for idx, value in enumerate(sorted_values)}
			data[col] = data[col].map(mapping)

	if gaussian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	# sample balance
	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	# retained_features = correlation_ret[correlation_ret >= retain_threshold].index.tolist()
	# new_cols = []
	# new_num_cols = 0
	# for idx, feature in enumerate(data.columns.tolist()):
	#     if feature in retained_features:
	#         new_cols.append(feature)
	#         if idx < num_cols:
	#             new_num_cols += 1

	# data = data[new_cols]

	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_default_credit(normalize=True, verbose=False, threshold=None):

	if threshold is None:
		threshold = 0.1

	retain_threshold = 0.0

	data = pd.read_csv("./data/default_credit/default_creidt.csv")
	data = data.dropna()
	data = data.drop('ID', axis=1)

	# target
	target_col = 'default payment next month'

	# convert categorical to numerical
	# for col in ["SEX", "MARRIAGE", "EDUCATION"]:
	# 	values = data[col].value_counts().index.tolist()
	# 	corr_y = []
	# 	for value in values:
	# 		corr_y_data = data[data[col] == value][target_col].value_counts(normalize=True)
	# 		if corr_y_data.shape[0] == 1:
	# 			if corr_y_data.index[0] == 0:
	# 				corr = 0
	# 			else:
	# 				corr = 1e5
	# 		else:
	# 			corr = corr_y_data[0]/corr_y_data[1]

	# 		corr_y.append(corr)

	# 	sorted_values = sorted(values, key = lambda x: corr_y[values.index(x)])
	# 	np.random.seed(0)
	# 	np.random.shuffle(sorted_values)
	# 	mapping = {value: idx for idx, value in enumerate(sorted_values)}
	# 	print(mapping)
	# 	data[col] = data[col].map(mapping)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	# sample balance
	data_y0 = data[data[target_col] == 0]
	data_y1 = data[data[target_col] == 1]
	data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
	data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	retained_features = correlation_ret[correlation_ret >= retain_threshold].index.tolist()
	new_cols = []
	for idx, feature in enumerate(data.columns.tolist()):
		if feature in retained_features:
			new_cols.append(feature)

	data = data[new_cols]

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_firewall(normalize=True, verbose=False, threshold=None):
	threshold = 0.1
	if threshold is None:
		threshold = 0.1

	data = pd.read_csv("./data/firewall/log2.csv")
	data = data[data['Action'] != 'reset-both']
	target_col = 'Action'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	source_freq = data['Source Port'].value_counts(normalize=True)
	source_freq.name = 'source_freq'
	destination_freq = data['Destination Port'].value_counts(normalize=True)
	destination_freq.name = 'destination_freq'
	nat_source_freq = data['NAT Source Port'].value_counts(normalize=True)
	nat_source_freq.name = 'nat_source_freq'
	nat_destination_freq = data['NAT Destination Port'].value_counts(normalize=True)
	nat_destination_freq.name = 'nat_destination_freq'

	data = data.merge(source_freq, how='left', left_on='Source Port', right_index=True)
	data = data.merge(destination_freq, how='left', left_on='Destination Port', right_index=True)
	data = data.merge(nat_source_freq, how='left', left_on='NAT Source Port', right_index=True)
	data = data.merge(nat_destination_freq, how='left', left_on='NAT Destination Port', right_index=True)
	# data['sd_pair'] = data[['Source Port', 'Destination Port']].apply(lambda x: tuple(x), axis=1)
	# data['nat_sd_pair'] = data[['NAT Source Port', 'NAT Destination Port']].apply(lambda x: tuple(x), axis=1)
	# sd_pair_freq = data['sd_pair'].value_counts(normalize=True)
	# sd_pair_freq.name = 'sd_pair_freq'
	# nat_sd_pair_freq = data['nat_sd_pair'].value_counts(normalize=True)
	# nat_sd_pair_freq.name = 'nat_sd_pair_freq'
	# data = data.merge(sd_pair_freq, how = 'left', left_on='sd_pair', right_index=True)
	# data = data.merge(nat_sd_pair_freq, how = 'left', left_on='nat_sd_pair', right_index=True)

	data = data.drop(
		[
			'Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port', 'nat_source_freq'
		], axis=1
	)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	# # # sample balance
	# # data_y0 = data[data[target_col] == 0]
	# # data_y1 = data[data[target_col] == 1]
	# # data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
	# # data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_dry_bean(normalize=True, verbose=False, threshold=None, guassian=False):
	normalize = True
	if threshold is None:
		threshold = 0.1

	data = pd.read_excel("./data/dry_bean/Dry_Bean_Dataset.xlsx")
	target_col = 'Class'
	# data = data[data['Action'] != 'reset-both']
	# target_col = 'Action'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	data = data.drop(['Extent', 'Solidity', 'ShapeFactor4', 'roundness'], axis=1)

	if guassian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config

def process_bank_market(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/bank_market/bank-full.csv", sep=';')
	cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
	target_col = 'y'
	if pca:
		data = pd.get_dummies(data, columns=cat_cols)
		print(data.shape)
		pca = PCA(n_components=10)
		pca.fit(data.drop(target_col, axis=1))
		data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)
	else:
		for col in cat_cols:
			cats = data[col].value_counts().index.tolist()
			np.random.seed(0)
			np.random.shuffle(cats)
			mapping = {cat: idx for idx, cat in enumerate(cats)}
			data[col] = data[col].map(mapping)

	# target_col = 'Y'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	if gaussian:
		data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	# sample balance
	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_ijcnn(normalize=True, verbose=False, threshold=None, sample=False, pca=False, gaussian=False):
	if threshold is None:
		threshold = 0.1

	data_obj = fetch_openml(data_id=1575, as_frame='auto', parser='auto')
	X = pd.DataFrame(data_obj.data.todense(), columns=data_obj.feature_names)
	y = pd.DataFrame(data_obj.target, columns=data_obj.target_names)
	data = pd.concat([X, y], axis=1)

	target_col = 'class'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()
	data = move_target_to_end(data, target_col)

	if pca:
		category_cols = [data.columns[idx] for idx in list(range(0, 8))]

		pca = PCA(n_components=20)
		pca.fit(data.iloc[:, :-1])
		data_pca = pca.transform(data.iloc[:, :-1])
		data_pca = pd.DataFrame(data_pca)
		data_pca = pd.concat([data_pca, data[target_col]], axis=1)
		data = data_pca

	if gaussian:
		data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	# # # sample balance
	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config



def process_svm(normalize=True, verbose=False, threshold=None, gaussian=False):
	if threshold is None:
		threshold = 0.1
	data_train = pd.read_csv("./data/svm1/svm_p.csv", sep=',', header=None)
	data_test = pd.read_csv("./data/svm1/svm_pt.csv", sep=',', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '0'
	data[target_col] = pd.factorize(data[target_col])[0]
	if gaussian:
		data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_pendigits(normalize=True, verbose=False, threshold=None, gaussian=False):
	if threshold is None:
		threshold = 0.1

	data_train = pd.read_csv("./data/pendigits/pendigits.tra", sep=',', header=None)
	data_test = pd.read_csv("./data/pendigits/pendigits.tes", sep=',', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)
	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '16'
	data[target_col] = pd.factorize(data[target_col])[0]

	if gaussian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_statlog(normalize=True, verbose=False, threshold=None, pca=False, gaussian=False):
	if threshold is None:
		threshold = 0.1
	data_train = pd.read_csv("./data/statlog/shuttle.trn.trn", sep='\s+', header=None)
	data_test = pd.read_csv("./data/statlog/shuttle.tst.tst", sep='\s+', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)

	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '9'
	data[target_col] = pd.factorize(data[target_col])[0]

	# sample balance
	data = data[data[target_col].isin([2, 1, 3])]
	data_rest = data[data[target_col].isin([1, 3])]
	data_more = data[data[target_col].isin([2])]
	data_more = data_more.sample(n=data_rest.shape[0], random_state=42)
	data = pd.concat([data_rest, data_more], axis=0).reset_index(drop=True)

	if pca:
		pca = PCA(n_components=10)
		pca.fit(data.drop(target_col, axis=1))
		data = pd.concat([pd.DataFrame(pca.transform(data.drop(target_col, axis=1))), data[target_col]], axis=1)

	if gaussian:
		data = convert_gaussian(data, target_col)

	data[target_col] = pd.factorize(data[target_col])[0]

	if normalize:
		data = normalization(data, target_col)

	# #data = convert_gaussian(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_avila(normalize=True, verbose=False, threshold=None):

	if threshold is None:
		threshold = 0.1

	data_train = pd.read_csv("./data/avila/avila-tr.txt", sep=',', header=None)
	data_test = pd.read_csv("./data/avila/avila-ts.txt", sep=',', header=None)
	data = pd.concat([data_train, data_test], axis=0).reset_index(drop=True)

	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '10'
	data = data[data[target_col].isin(['A', 'F', 'E', 'I', 'X', 'H', 'G', 'D'])]
	data[target_col] = pd.factorize(data[target_col])[0]

	# if normalize:
	# 	data = normalization(data, target_col)

	# #data = convert_gaussian(data, target_col)
	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
#
# Regression
#
########################################################################################################################

########################################################################################################################
# Diabetes
########################################################################################################################
def process_diabetes(normalize=True, verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.15
	from sklearn.datasets import load_diabetes
	data_obj = load_diabetes(as_frame=True)
	data = data_obj['frame']
	data = data.dropna()
	target_col = 'target'

	# # move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
	# # correlation
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'important_features': important_features,
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		'num_cols': data.shape[1] - 1,
		'task_type': 'regression',
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# California Housing
########################################################################################################################
def process_california_housing(normalize=True, verbose=False, threshold=0.1):
	if threshold is None:
		threshold = 0.1
	from sklearn.datasets import fetch_california_housing
	data_obj = fetch_california_housing(data_home='./data/california_housing', as_frame=True)
	data = data_obj['frame']
	sample_size = 20000
	data = data.sample(sample_size, random_state=42)
	data = data.dropna()
	target_col = 'MedHouseVal'

	if normalize:
		data = normalization(data, target_col)

	# # move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
	# # correlation
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'important_features': important_features,
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		'num_cols': data.shape[1] - 1,
		'task_type': 'regression',
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# Housing
########################################################################################################################
def process_housing(normalize=True, verbose=False, threshold=0.2):
	if threshold is None:
		threshold = 0.2

	data = pd.read_csv("./data/housing/housing.csv", delimiter='\s+', header=None)
	data.columns = [str(i) for i in range(1, data.shape[1] + 1)]
	# sample_size = 1000
	# data = data.sample(sample_size, random_state=42)
	data = data.dropna()
	# data
	target_col = '14'

	if normalize:
		data = normalization(data, target_col)

	# # # move target to the end of the dataframe
	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
	# # # correlation
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'important_features': important_features,
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		'num_cols': data.shape[1] - 1,
		'task_type': 'regression',
	}
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# red wine regression
########################################################################################################################
def process_red_reg(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-red.csv", delimiter=';')
	# data = data.drop(["id", "Unnamed: 32"], axis=1)
	data = data.dropna()
	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)

	# print("Wine Red data loaded. Train size {}, Test size {}".format(train.shape, test.shape))
	correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	print(important_features)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'important_features': important_features,
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'regression',
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


########################################################################################################################
# red wine regression
########################################################################################################################
def process_white_reg(normalize=True, verbose=False, threshold=0.15):
	if threshold is None:
		threshold = 0.15
	data = pd.read_csv("./data/wine/winequality-white.csv", delimiter=';')
	# data = data.drop(["id", "Unnamed: 32"], axis=1)
	data = data.dropna()
	target_col = 'quality'
	if normalize:
		data = normalization(data, target_col)
	data = move_target_to_end(data, target_col)

	# print("Wine Red data loaded. Train size {}, Test size {}".format(train.shape, test.shape))
	correlation_ret = data.corrwith(data[target_col]).abs().sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)
	print(important_features)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'important_features': important_features,
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'regression',
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	return data, data_config


def process_susy(normalize=True, verbose=False, threshold=0.15, gaussian=False):
	if threshold is None:
		threshold = 0.15

	data = pd.read_csv("./data/susy/SUSY.csv", sep=',', header=None)

	data = data.dropna()
	data.columns = [str(i) for i in range(data.shape[1])]
	target_col = '0'
	data[target_col] = pd.factorize(data[target_col])[0]
	data[target_col].value_counts()

	data = data.sample(frac=0.01, random_state=42)
	if gaussian:
		data = convert_gaussian(data, target_col)
	if normalize:
		data = normalization(data, target_col)

	# #data = convert_gaussian(data, target_col)
	data = move_target_to_end(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


def process_higgs(verbose=False, threshold=0.15):

	if threshold is None:
		threshold = 0.15

	data = pd.read_csv("./data/higgs/higgs_new.csv", sep=',')

	data = data.dropna()
	target_col = '0'
	# #data = convert_gaussian(data, target_col)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	data_config = {
		'target': target_col,
		'important_features_idx': [data.columns.tolist().index(feature) for feature in important_features],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	return data, data_config


########################################################################################################################
# factory function to load dataset
########################################################################################################################
def load_data(dataset_name, normalize=True, verbose=False, threshold=None):

	########################################################################################################################
	# Classification
	########################################################################################################################
	if dataset_name == 'iris':
		return process_iris(normalize, verbose, threshold)
	elif dataset_name == 'breast':
		return process_breast(normalize, verbose, threshold)
	elif dataset_name == 'ecoli':
		return process_ecoli(normalize, verbose, threshold)
	elif dataset_name == 'white':
		return process_white(normalize, verbose, threshold)
	elif dataset_name == 'red':
		return process_red(normalize, verbose, threshold)
	elif dataset_name == 'wine_quality_all':
		return process_wine_three(normalize, verbose, threshold)
	elif dataset_name == 'spam':
		return process_spam(normalize, verbose, threshold)
	elif dataset_name == 'blocks':
		return process_blocks(normalize, verbose, threshold)
	elif dataset_name == 'glass':
		return process_glass(normalize, verbose, threshold)
	elif dataset_name == 'optdigits':
		return process_optdigits(normalize, verbose, threshold)
	elif dataset_name == 'segmentation':
		return process_segmentation(normalize, verbose, threshold)
	elif dataset_name == 'sonar':
		return process_sonar(normalize, verbose, threshold)
	elif dataset_name == 'sensor':
		return process_sensor(normalize, verbose, threshold)
	elif dataset_name == 'sensor_pca':
		return process_sensor(normalize, verbose, threshold, pca=True)
	elif dataset_name == 'waveform':
		return process_waveform(normalize, verbose, threshold)
	elif dataset_name == 'yeast':
		return process_yeast(normalize, verbose, threshold)
	elif dataset_name == 'letter':
		return process_letter(normalize, verbose, threshold)
	elif dataset_name == 'raisin':
		return process_raisin(normalize, verbose, threshold)
	elif dataset_name == 'dermatology':
		return process_dermatology(normalize, verbose, threshold)
	elif dataset_name == "pima_diabetes":
		return process_pima_diabetes(normalize, verbose, threshold)
	elif dataset_name == 'telugu_vowel':
		return process_telugu_vowel(normalize, verbose, threshold)
	elif dataset_name == 'telugu_tabular':
		return process_telugu_tabular(normalize, verbose, threshold)
	elif dataset_name == 'wine':
		return process_wine(normalize, verbose, threshold)
	elif dataset_name == 'wifi':
		return process_wifi(normalize, verbose, threshold)
	elif dataset_name == 'adult':
		return process_adult(normalize, verbose, threshold)
	elif dataset_name == 'adult_pca':
		return process_adult(normalize, verbose, threshold, sample=False, pca=True, gaussian=True)
	elif dataset_name == 'adult_balanced':
		return process_adult(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'adult_balanced_pca':
		return process_adult(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
	elif dataset_name == 'default_credit':
		return process_default_credit(normalize, verbose, threshold)
	elif dataset_name == 'firewall':
		return process_firewall(normalize, verbose, threshold)
	elif dataset_name == 'dry_bean':
		return process_dry_bean(normalize, verbose, threshold)
	elif dataset_name == 'dry_bean_g':
		return process_dry_bean(normalize, verbose, threshold, guassian=True)
	elif dataset_name == 'bank_marketing':
		return process_bank_market(normalize, verbose, threshold)
	elif dataset_name == 'bank_marketing_balanced':
		return process_bank_market(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'bank_balanced_pca':
		return process_bank_market(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
	elif dataset_name == 'ijcnn':
		return process_ijcnn(normalize, verbose, threshold)
	elif dataset_name == 'ijcnn_balanced':
		return process_ijcnn(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'ijcnn_balanced_pca':
		return process_ijcnn(normalize, verbose, threshold, sample=True, pca=True, gaussian=True)
	elif dataset_name == 'svm':
		return process_svm(normalize, verbose, threshold)
	elif dataset_name == 'svm_g':
		return process_svm(normalize, verbose, threshold, gaussian=True)
	elif dataset_name == 'pendigits':
		return process_pendigits(normalize, verbose, threshold)
	elif dataset_name == 'pendigits_g':
		return process_pendigits(normalize, verbose, threshold, gaussian=True)
	elif dataset_name == 'statlog':
		return process_statlog(normalize, verbose, threshold)
	elif dataset_name == 'statlog_pca':
		return process_statlog(normalize, verbose, threshold, pca=True, gaussian=True)
	elif dataset_name == 'avila':
		return process_avila(normalize, verbose, threshold)
	elif dataset_name == 'susy':
		return process_susy(normalize, verbose, threshold)
	elif dataset_name == 'susy_g':
		return process_susy(normalize, verbose, threshold, gaussian=True)
	elif dataset_name == 'higgs':
		return process_higgs(verbose, threshold)
	#######################################################################################################################
	# Healthcare Dataset
	#######################################################################################################################
	elif dataset_name == 'nhis_income':
		return process_NHIS_income(pca=False)
	elif dataset_name == 'nhis_income_pca':
		return process_NHIS_income(pca=True)
	elif dataset_name == 'skin':
		return process_skin(normalize, verbose, threshold, sample=False)
	elif dataset_name == 'skin_balanced':
		return process_skin(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'sepsis':
		return process_sepsis(verbose, threshold)
	elif dataset_name == 'diabetic':
		return process_diabetic(verbose, threshold)
	elif dataset_name == 'diabetic_balanced':
		return process_diabetic(verbose, threshold, sample=True)
	elif dataset_name == 'diabetic2':
		return process_diabetic2(verbose, threshold, sample=True)
	elif dataset_name == 'cardio':
		return process_cardio(verbose, threshold)
	elif dataset_name == 'genetic_balanced':
		return process_genetic(sample=True)
	#######################################################################################################################
	# Regression
	#######################################################################################################################
	elif dataset_name == 'diabetes':
		return process_diabetes(normalize, verbose, threshold)
	elif dataset_name == 'california_housing':
		return process_california_housing(normalize, verbose, threshold)
	elif dataset_name == 'housing':
		return process_housing(normalize, verbose, threshold)
	elif dataset_name == 'red_reg':
		return process_red_reg(normalize, verbose, threshold)
	elif dataset_name == 'white_reg':
		return process_white_reg(normalize, verbose, threshold)
	#######################################################################################################################
	# Used Dataset
	#######################################################################################################################
	elif dataset_name == 'mimiciii_mo':
		return process_mimiciii_mortality()
	elif dataset_name == 'mimiciii_icd':
		return process_mimic_icd2()
	elif dataset_name == 'mimiciii_mo2':
		return process_mimic_mo()
	elif dataset_name == 'mimiciii_los':
		return process_mimic_los()
	elif dataset_name == 'genetic':
		return process_genetic(sample=False)
	elif dataset_name == 'heart':
		return process_heart(pca=True, sample=False)
	elif dataset_name == 'heart_balanced':
		return process_heart(pca=True, sample=True)
	elif dataset_name == 'codrna':
		return process_codrna(normalize, verbose, threshold, sample=False)
	elif dataset_name == 'codrna_balanced':
		return process_codrna(normalize, verbose, threshold, sample=True)
	elif dataset_name == 'codon':
		return process_codon(verbose, threshold)
	elif dataset_name == 'hhp_ct1':
		return process_hhp(version = 'ct_np1')
	elif dataset_name == 'hhp_ct2':
		return process_hhp(version = 'ct_np6')
	elif dataset_name == 'hhp_ct3':
		return process_hhp(version = 'ct_np9')
	elif dataset_name == 'vehicle':
		return process_vehicle()
	elif dataset_name == 'vehicle2':
		return process_vehicle2()
	elif dataset_name == 'eicu_mo1':
		return process_eicu(version = 'mo_np1120')
	elif dataset_name == 'heart_disease_binary':
		return process_heart_disease(version = 'binary')
	elif dataset_name == 'hhp_los_np1':
		return process_hhp(version = 'los_np1')
	elif dataset_name == 'heart_disease_binary':
		return process_heart_disease(version = 'binary')
	elif dataset_name == 'heart_disease_binary1':
		return process_heart_disease(version = 'binary1')
	elif dataset_name == 'vehicle':
		return process_vehicle()
	elif dataset_name == 'eicu_mo1':
		return process_eicu(version = 'mo_np1120')
	elif dataset_name == 'eicu_mo2':
		return process_eicu(version = 'mo_np1121')
	elif dataset_name == 'eicu_mo3':
		return process_eicu(version = 'mo_np1122')
	elif dataset_name == 'eicu_mo_np1':
		return process_eicu(version = 'mo_np22')
	elif dataset_name == 'eicu_mo_np2':
		return process_eicu(version = 'mo_np8')
	elif dataset_name == 'eicu_mo_np3':
		return process_eicu(version = 'mo_np4')
	elif dataset_name == 'eicu_mo_np4':
		return process_eicu(version = 'mo_np10')
	else:
		raise Exception("Unknown dataset name {}".format(dataset_name))

def process_hhp(version, verbose=True):
	data = pd.read_csv(f'./data/hhp/data_clean_{version}.csv')
	data_config = json.load(open(f'./data/hhp/data_config_{version}.json'))
	
	if verbose:
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	if 'client_split_indices' in data_config:
		data_arrays = np.array_split(data.values, data_config['client_split_indices'])
		data = [pd.DataFrame(data_array, columns=data.columns) for data_array in data_arrays]

	return data, data_config

def process_heart_disease(version, verbose=True):
	data = pd.read_csv(f'./data/heart_disease/data_clean_{version}.csv')
	data_config = json.load(open(f'./data/heart_disease/data_config_{version}.json'))
	
	if verbose:
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	if 'client_split_indices' in data_config:
		data_arrays = np.array_split(data.values, data_config['client_split_indices'])
		data = [pd.DataFrame(data_array, columns=data.columns) for data_array in data_arrays]
		print([data_array.shape for data_array in data])
	return data, data_config

def process_vehicle(verbose = True):
	data = pd.read_csv(f'./data/vehicle/data_clean.csv')
	data_config = json.load(open(f'./data/vehicle/data_config.json'))

	if verbose:
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	if 'client_split_indices' in data_config:
		data_arrays = np.array_split(data.values, data_config['client_split_indices'])
		data = [pd.DataFrame(data_array, columns=data.columns) for data_array in data_arrays]
		print([data_array.shape for data_array in data])
	return data, data_config

def process_vehicle2(verbose = True):
	data = pd.read_csv(f'./data/vehicle/data_clean3.csv')
	data_config = json.load(open(f'./data/vehicle/data_config3.json'))

	if verbose:
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	if 'client_split_indices' in data_config:
		data_arrays = np.array_split(data.values, data_config['client_split_indices'])
		data = [pd.DataFrame(data_array, columns=data.columns) for data_array in data_arrays]
		print([data_array.shape for data_array in data])
	return data, data_config


def process_eicu(version, verbose=True):
	data = pd.read_csv(f'./data/eicu/data_clean_{version}.csv')
	data_config = json.load(open(f'./data/eicu/data_config_{version}.json'))

	if verbose:
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)

	if 'client_split_indices' in data_config:
		data_arrays = np.array_split(data.values, data_config['client_split_indices'])
		data = [pd.DataFrame(data_array, columns=data.columns) for data_array in data_arrays]

	return data, data_config

if __name__ == '__main__':
	process_breast()
