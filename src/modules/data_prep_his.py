import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer, LabelEncoder
from dython.nominal import correlation_ratio
from loguru import logger
from sklearn.datasets import fetch_openml
import numpy as np
from .data_prep_utils import (
	convert_gaussian, normalization, drop_unique_cols, one_hot_categorical, move_target_to_end,
)
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler


def process_NHIS_income(
		normalize=True, verbose=False, threshold=None, pca=True
):
	if threshold is None:
		threshold = 0.1
	data_folder = 'data/NHIS/'
	paths = {
		'family': data_folder + 'familyxx.csv',
		'child': data_folder + 'samchild.csv',
		'adult': data_folder + 'samadult.csv',
		'person': data_folder + 'personsx.csv',
		'household': data_folder + 'househld.csv',
		'injury': data_folder + 'injpoiep.csv'
	}
	sep = ','

	household = pd.read_csv(paths['household'], sep=sep)
	adult = pd.read_csv(paths['adult'], sep=sep)
	family = pd.read_csv(paths['family'], sep=sep)
	person = pd.read_csv(paths['person'], sep=sep)

	# Merge dataframes
	df = household.merge(
		family, how='inner', on=['SRVY_YR', 'HHX'],
		suffixes=('', '%to_drop')
	).merge(
		person, how='inner', on=['SRVY_YR', 'HHX', 'FMX'],
		suffixes=('', '%to_drop')
	).merge(
		adult, how='inner', on=['SRVY_YR', 'HHX', 'FMX', 'FPX'],
		suffixes=('', '%to_drop')
	).dropna(subset=['ERNYR_P'])

	df = df.loc[:, ~df.columns.str.endswith('%to_drop')]
	df['IDX'] = df.index
	df.head()
	print("Shape of raw data: ", df.shape)
	df = df.reset_index(drop=True)

	###########################################################################
	# Target
	###########################################################################
	target_col = 'ERNYR_P'
	df[target_col] = df[target_col].map(lambda x: x if x < 90 else np.nan)
	df.dropna(subset=['ERNYR_P'], inplace=True)
	df[target_col] = df[target_col].map(lambda x: 1 if x > 6 else 0)
	df = move_target_to_end(df, target_col)
	df[target_col] = pd.factorize(df[target_col])[0]

	###########################################################################
	# drop columns with unique values and many values
	###########################################################################
	unique_cols = []
	for column in adult.columns:
		if df[column].nunique() == 1:
			unique_cols.append(column)

	df.drop(unique_cols, axis=1, inplace=True)
	print("Shape of data after dropping columns only contains one value: ", df.shape)

	# drop columns with too many values
	many_values_cols = []
	for column in df.columns:
		if df[column].nunique() / df.shape[0] > 0.7:
			many_values_cols.append(column)

	df.drop(many_values_cols, axis=1, inplace=True)
	print("Shape of data after dropping high cardinality columns: ", df.shape)

	###########################################################################
	# drop missing values
	###########################################################################
	ms_pct = df.isnull().sum() / adult.shape[0]
	ms_thres = 0.0
	ms_cols = ms_pct[ms_pct > ms_thres].index.tolist()
	df = df.drop(columns=ms_cols)
	df = df.dropna(how='any')
	print("Shape of data remove columns with missing values: ", df.shape)

	# numerical columns
	numerical_cols = [col for col in df.columns if col.startswith('WT')]
	print(numerical_cols)

	###########################################################################
	# Feature selection
	###########################################################################
	cat_cols = [col for col in df.columns if col not in numerical_cols and col != target_col]
	mi = mutual_info_classif(
		X=df[cat_cols], y=df[target_col], random_state=42, discrete_features=True
	)

	num_features = 10
	corr_mi = pd.Series(mi)
	corr_mi.index = cat_cols
	features = corr_mi.sort_values(ascending=False)[0:num_features].index.tolist()
	print(features)
	###########################################################################
	# one-hot encoding
	###########################################################################
	oh = OneHotEncoder(
		sparse_output=False, drop='first', max_categories=15, handle_unknown='ignore'
	)
	X = df[features]
	X_cat = oh.fit_transform(X)
	X_cat = pd.DataFrame(X_cat)
	print(X_cat.shape)

	###########################################################################
	# final df
	###########################################################################
	y = df[target_col].reset_index(drop=True)
	X_num = df[numerical_cols].reset_index(drop=True)
	print(X_cat.shape, X_num.shape, y.shape)
	df_new = pd.concat([X_cat, y], axis=1)
	df_new.columns = df_new.columns.astype(str)
	print("Shape of data after one-hot encoding: ", df_new.shape)

	###########################################################################
	# PCA
	###########################################################################
	if pca:
		pca_trans = PCA(n_components=30, random_state=42, svd_solver='auto')
		X = df_new.drop([target_col], axis=1)
		X_pca = pca_trans.fit_transform(X)
		print("Shape of data after PCA:", X_pca.shape)

		df_new = pd.concat([pd.DataFrame(X_pca), df_new[target_col]], axis=1)

		###########################################################################
		# Normalization
		###########################################################################
		df_new = convert_gaussian(df_new, target_col)
		df_new = normalization(df_new, target_col)
		data = df_new.copy()

		###########################################################################
		# Correlation
		###########################################################################
		threshold = 0.1
		correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
		important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
		important_features.remove(target_col)

	else:
		df_new = normalization(df_new, target_col)
		data = df_new.copy()

		###########################################################################
		# Correlation
		###########################################################################
		threshold = 0.1
		correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
		important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
		important_features.remove(target_col)
		data = data[important_features + [target_col]]
		data = data.drop(['25', '3'], axis=1)

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


def process_heart(verbose=False, threshold=None, pca=False):

	if threshold is None:
		threshold = 0.1

	df = pd.read_csv('./data/heart/heart_2020_cleaned.csv')
	target_col = 'HeartDisease'
	df[target_col].value_counts()
	df = move_target_to_end(df, target_col)

	if pca:
		categorical_columns = [
			'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'DiffWalking',
			'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
			'Asthma', 'KidneyDisease', 'SkinCancer'
		]

		numerical_columns = [col for col in df.columns if col not in categorical_columns and col != target_col]
		robust_scaler = MinMaxScaler()
		oh_encoder = OneHotEncoder(max_categories=10, drop='first', sparse_output=False)

		X_cat = oh_encoder.fit_transform(df[categorical_columns])
		print(X_cat.shape)
		X_num = robust_scaler.fit_transform(df[numerical_columns])
		print(X_num.shape)
		X = np.concatenate([X_num, X_cat], axis=1)

	# # label=LabelEncoder()
	# # for col in df:
	# #     df[col]=label.fit_transform(df[col])
		under = RandomUnderSampler(random_state=42)
		X_new, y_new = under.fit_resample(X, df[target_col].values)

		# pca
		pca = PCA(n_components=0.90)
		X_pca = pca.fit_transform(X_new)
		print(X_pca.shape)

		df_new = pd.DataFrame(np.concatenate([X_pca, y_new.reshape(-1, 1)], axis=1))
		df_new.columns = df_new.columns.astype(str)

		target_col = '25'
		df_new = move_target_to_end(df_new, target_col)
		df_new[target_col] = pd.factorize(df_new[target_col])[0]
		df_new = convert_gaussian(df_new, target_col)
		df_new = normalization(df_new, target_col)
		print(df_new.shape)

		data = df_new

	else:
		# label=LabelEncoder()
		# for col in df:
		# 	df[col]=label.fit_transform(df[col])
		raise NotImplementedError

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
