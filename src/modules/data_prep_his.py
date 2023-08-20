import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, PowerTransformer, LabelEncoder, StandardScaler, RobustScaler
)
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


def process_heart(verbose=False, threshold=None, pca=False, sample = False):

	if threshold is None:
		threshold = 0.1
	df = pd.read_csv('./data/heart/heart_2020_cleaned.csv')
	target_col = 'HeartDisease'
	df['HeartDisease'] = df['HeartDisease'].map({"No": 0, "Yes": 1})
	df[target_col].value_counts()
	df = move_target_to_end(df, target_col)

	categorical_columns = [
		'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
		'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 
		'Asthma', 'KidneyDisease', 'SkinCancer'
	]

	numerical_cols = [col for col in df.columns if col not in categorical_columns and col != target_col]
	print(numerical_cols)
	oh_encoder = OneHotEncoder(max_categories=10, drop='first', sparse_output = False)
	X_cat = oh_encoder.fit_transform(df[categorical_columns])
	X_num = df[numerical_cols].values
	robust_scaler = MinMaxScaler()
	X_num = robust_scaler.fit_transform(df[numerical_cols])
	X = np.concatenate([X_num, X_cat], axis=1)

	# pca
	pca = PCA(n_components=0.95)
	X_new = pca.fit_transform(X)
	y_new = df[target_col].values

	# new dataframe
	df_new = pd.DataFrame(np.concatenate([X_new, y_new.reshape(-1, 1)], axis=1))
	df_new.columns = df_new.columns.astype(str)

	target_col = df_new.columns[-1]
	df_new = move_target_to_end(df_new, target_col)
	df_new[target_col] = pd.factorize(df_new[target_col])[0]
	df_new = convert_gaussian(df_new, target_col)
	df_new = normalization(df_new, target_col)
	print(df_new.shape)

	data = df_new

	if sample:
		# sampling
		target_col = data.columns[-1]
		print(data.shape)
		under = RandomUnderSampler(random_state=42)
		X = data.drop([target_col], axis=1).values
		X_new, y_new = under.fit_resample(X, data[target_col].values)
		data = pd.DataFrame(np.concatenate([X_new, y_new.reshape(-1, 1)], axis = 1))
		print(data.shape)
		target_col = data.columns[-1]
		df0 = data[data[target_col] == 0]
		df1 = data[data[target_col] == 1]
		if df0.shape[0] > df1.shape[0]:
			df0 = df0.sample(n=df1.shape[0], random_state=42)
		elif df0.shape[0] < df1.shape[0]:
			df1 = df1.sample(n=df0.shape[0], random_state=42)
		
		data = pd.concat([df0, df1], axis=0)
		print(data.shape)
	
	if len(data) >= 20000:
		data = data.sample(n = 20000, random_state=42).reset_index(drop=True)

	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	print(data.shape)

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


def process_codrna(normalize=True, verbose=False, threshold=None, sample=True, gaussian=True):

	if threshold is None:
		threshold = 0.1

	data_obj = fetch_openml(data_id=351, as_frame='auto', parser='auto')
	X = pd.DataFrame(data_obj.data.todense(), columns=data_obj.feature_names)
	y = pd.DataFrame(data_obj.target, columns=data_obj.target_names)
	data = pd.concat([X, y], axis=1)

	target_col = 'Y'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()

	if gaussian:
		data = convert_gaussian(data, target_col)

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)
	correlation_ret = data.corrwith(data[target_col], method=correlation_ratio).sort_values(ascending=False)
	important_features = correlation_ret[correlation_ret >= threshold].index.tolist()
	important_features.remove(target_col)

	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		if data_y0.shape[0] > data_y1.shape[0]:
			data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		elif data_y0.shape[0] < data_y1.shape[0]:
			data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)
	
	if len(data) >= 20000:
		data = data.sample(n=20000).reset_index(drop=True)

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


def process_skin(normalize=True, verbose=False, threshold=None, sample=False):
	if threshold is None:
		threshold = 0.1

	data_obj = fetch_openml(data_id=1502, as_frame='auto', parser='auto')
	X = pd.DataFrame(data_obj.data, columns=data_obj.feature_names)
	y = pd.DataFrame(data_obj.target, columns=data_obj.target_names)
	data = pd.concat([X, y], axis=1)

	target_col = 'Class'
	data[target_col] = pd.factorize(data[target_col])[0]
	data = data.dropna()

	if normalize:
		data = normalization(data, target_col)

	data = move_target_to_end(data, target_col)

	# # # sample balance
	if sample:
		data_y0 = data[data[target_col] == 0]
		data_y1 = data[data[target_col] == 1]
		if data_y0.shape[0] > data_y1.shape[0]:
			data_y0 = data_y0.sample(n=data_y1.shape[0], random_state=0)
		elif data_y0.shape[0] < data_y1.shape[0]:
			data_y1 = data_y1.sample(n=data_y0.shape[0], random_state=0)
		data = pd.concat([data_y0, data_y1], axis=0).reset_index(drop=True)
	
	if len(data) >= 20000:
		data = data.sample(n = 20000).reset_index(drop=True)

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


def process_codon(verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv("./data/codon/codon_usage.csv", sep=',', low_memory=False)

	data = data.dropna()
	#data.columns = [str(i) for i in range(data.sh
	data = data.drop(['SpeciesID', 'Ncodons', 'SpeciesName', 'DNAtype'], axis=1)
	target_col = 'Kingdom'
	data = data[data[target_col] != 'plm']
	data[target_col], codes = pd.factorize(data[target_col])
	data = move_target_to_end(data, target_col)
	data = normalization(data, target_col)

	# pca
	pca = PCA(n_components=0.9)
	X = pca.fit_transform(data.drop(target_col, axis=1).values)
	y = data[target_col].values

	# new dataframe
	df_new = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
	df_new.columns = df_new.columns.astype(str)

	target_col = df_new.columns[-1]
	data = df_new
	print(data.shape)

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


def process_sepsis(verbose=False, threshold=None):
	if threshold is None:
		threshold = 0.1
	data = pd.read_csv('./data/sepsis/sepsis_survival_primary_cohort.csv')
	data = data.dropna()
	target_col = 'hospital_outcome_1alive_0dead'
	data[target_col] = data[target_col].astype(int)
	#data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	# sample balanced
	# data0 = data[data[target_col] == 0]
	# data1 = data[data[target_col] == 1]
	# if data0.shape[0] > data1.shape[0]:
	#     data0 = data0.sample(data1.shape[0])
	# else:
	#     data1 = data1.sample(data0.shape[0])

	# data = pd.concat([data0, data1], axis=0)

	data = data.sample(n = 20000, random_state=42)

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


def process_diabetic(verbose=False, threshold=None, sample=False):

	if threshold is None:
		threshold = 0.1

	data = pd.read_csv('./data/diabetic/diabetic_data.csv')
	data = data.replace('?', np.nan)
	print(data.shape)
	target_col = 'readmitted'
	data[target_col] = data[target_col].map({'NO': 0, '>30': 0, '<30': 1})
	data = move_target_to_end(data, target_col)
	# data = data.drop_duplicates(subset= ['patient_nbr'], keep = 'first')

	# drop columns and handle missing values
	drop_cols = [
		'max_glu_serum', 'A1Cresult', 'weight', 'encounter_id', 'patient_nbr', 
		'examide', 'citoglipton'
	]
	data = data.drop(drop_cols, axis=1)
	data['payer_code'] = data['payer_code'].fillna('None')
	data['medical_specialty'] = data['medical_specialty'].fillna('None')
	data['diag_1'] = data['diag_1'].fillna('None')
	data['diag_2'] = data['diag_2'].fillna('None')
	data['diag_3'] = data['diag_3'].fillna('None')

	data = data[data['race'].isnull() == False]
	data = data[data['gender'].isin(['Male', 'Female'])]
	print(data.shape)

	num_cols = [
		'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
		'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 
		]

	cat_cols = [col for col in data.columns if col not in num_cols and col != target_col]

	# one-hot encoding
	oh_encoder = OneHotEncoder(
		sparse_output=False, handle_unknown='ignore', max_categories=100, drop='first'
	)

	X_cat = oh_encoder.fit_transform(data[cat_cols].values)
	pca = PCA(n_components=0.6)
	X_cat = pca.fit_transform(X_cat)
	print("X_cat:", X_cat.shape)

	# min-max scaling
	scaler = StandardScaler()
	X_num = scaler.fit_transform(data[num_cols].values)

	# combine
	X = np.concatenate([X_num, X_cat], axis=1)
	y = data[target_col].values

	# combine
	data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
	target_col = data.columns[-1]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	if sample:
		data0 = data[data[target_col] == 0]
		data1 = data[data[target_col] == 1]
		if data0.shape[0] > data1.shape[0]:
			data0 = data0.sample(data1.shape[0])
		else:
			data1 = data1.sample(data0.shape[0])
		
		data = pd.concat([data0, data1], axis=0)

	if data.shape[0] > 20000:
		data = data.sample(n = 20000, random_state=42)
	print(data.shape)

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

def process_diabetic2(verbose=False, threshold=None, sample=False):

	if threshold is None:
		threshold = 0.1

	data = pd.read_csv('./data/diabetic/diabetic_data.csv')
	data = data.replace('?', np.nan)
	print(data.shape)
	target_col = 'readmitted'
	data[target_col] = data[target_col].map({'NO': 2, '>30': 0, '<30': 1})
	data = data[data[target_col] != 2]
	data = move_target_to_end(data, target_col)
	# data = data.drop_duplicates(subset= ['patient_nbr'], keep = 'first')

	# drop columns and handle missing values
	drop_cols = [
		'max_glu_serum', 'A1Cresult', 'weight', 'encounter_id', 'patient_nbr', 
		'examide', 'citoglipton'
	]
	data = data.drop(drop_cols, axis=1)
	data['payer_code'] = data['payer_code'].fillna('None')
	data['medical_specialty'] = data['medical_specialty'].fillna('None')
	data['diag_1'] = data['diag_1'].fillna('None')
	data['diag_2'] = data['diag_2'].fillna('None')
	data['diag_3'] = data['diag_3'].fillna('None')

	data = data[data['race'].isnull() == False]
	data = data[data['gender'].isin(['Male', 'Female'])]
	print(data.shape)

	num_cols = [
		'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
		'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 
		]

	cat_cols = [col for col in data.columns if col not in num_cols and col != target_col]

	# one-hot encoding
	oh_encoder = OneHotEncoder(
		sparse_output=False, handle_unknown='ignore', max_categories=100, drop='first'
	)

	X_cat = oh_encoder.fit_transform(data[cat_cols].values)
	pca = PCA(n_components=0.6)
	X_cat = pca.fit_transform(X_cat)
	print("X_cat:", X_cat.shape)

	# min-max scaling
	scaler = StandardScaler()
	X_num = scaler.fit_transform(data[num_cols].values)

	# combine
	X = np.concatenate([X_num, X_cat], axis=1)
	y = data[target_col].values

	# combine
	data = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1))
	target_col = data.columns[-1]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	if sample:
		data0 = data[data[target_col] == 0]
		data1 = data[data[target_col] == 1]
		if data0.shape[0] > data1.shape[0]:
			data0 = data0.sample(data1.shape[0])
		else:
			data1 = data1.sample(data0.shape[0])
		
		data = pd.concat([data0, data1], axis=0)

	if data.shape[0] > 20000:
		data = data.sample(n = 20000, random_state=42)
	print(data.shape)

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


def process_cardio(verbose = False, threshold = None):
	if threshold is None:
		threshold = 0.1

	data = pd.read_csv("./data/cardio/cardio_train.csv", sep=";")
	data = data.drop("id", axis=1)
	target_col = "cardio"
	data = move_target_to_end(data, target_col)
	data["age"] = round(data["age"] / 365)

	###############################################################################
	# outliers and missing values
	def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
		quartile1 = dataframe[col_name].quantile(q1)
		quartile3 = dataframe[col_name].quantile(q3)
		interquantile_range = quartile3 - quartile1
		up_limit = quartile3 + 1.5 * interquantile_range
		low_limit = quartile1 - 1.5 * interquantile_range
		return low_limit, up_limit

	def replace_with_thresholds(dataframe, variable):
		low_limit, up_limit = outlier_thresholds(dataframe, variable)
		dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
		dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    
	num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

	for col in num_cols:
		replace_with_thresholds(data, col)

	###############################################################################
	# feature engineering
	data.loc[(data["age"] < 18), "NEW_AGE"] = "Young"
	data.loc[(data["age"] > 18) & (data["age"] < 56), "NEW_AGE"] = "Mature"
	data.loc[(data["age"] >= 56), "NEW_AGE"] = "Old"

	cols1 = data["weight"]
	cols2 = data["height"] / 100
	data["bmi"] = cols1 / (cols2 ** 2)
	data.loc[(data["bmi"] < 18.5), "NEW_BMI"] = "under"
	data.loc[(data["bmi"] >= 18.5) & (data["bmi"] <= 24.99) ,"NEW_BMI"] = "healthy"
	data.loc[(data["bmi"] >= 25) & (data["bmi"] <= 29.99) ,"NEW_BMI"]= "over"
	data.loc[(data["bmi"] >= 30), "NEW_BMI"] = "obese"

	data.loc[(data["ap_lo"])<=89, "BLOOD_PRESSURE"] = "normal"
	data.loc[(data["ap_lo"])>=90, "BLOOD_PRESSURE"] = "hyper"
	data.loc[(data["ap_hi"])<=120, "BLOOD_PRESSURE"] = "normal"
	data.loc[(data["ap_hi"])>120, "BLOOD_PRESSURE"] = "normal"
	data.loc[(data["ap_hi"])>=140, "BLOOD_PRESSURE"] = "hyper"

	###############################################################################
	#encoding
	rs = RobustScaler()
	data[num_cols] = rs.fit_transform(data[num_cols])

	def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
		dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
		return dataframe

	ohe_cols = [col for col in data.columns if 10 >= len(data[col].unique()) >= 2 and col != target_col]
	print(ohe_cols)
	data = one_hot_encoder(data, ohe_cols, drop_first=True)
	print(data.shape)

	###############################################################################
	# pca
	data = move_target_to_end(data, target_col)
	pca = PCA(n_components=0.99)
	X = pca.fit_transform(data.iloc[:, :-1].values)
	data  = pd.DataFrame(np.concatenate((X, data.iloc[:, -1:].values), axis=1))
	print(data.shape)
	target_col = data.columns[-1]

	###############################################################################
	# scaling
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	if data.shape[0] > 20000:
		data = data.sample(n = 20000, random_state=42)

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

	print(data.shape)
	print(data[target_col].value_counts())
	
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	
	return data, data_config



def process_mimiciii_mortality():
	
	# read data
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	print(df_prescriptions.shape)

	# patient admission table
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)
	print(df_patient_admission.shape)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()
	target_col = 'mortality'

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)

	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 
	      'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	print(df_patient_admission.shape)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())
	df_patient_admission = pd.concat([df_patient_admission.drop(cat_cols, axis=1).reset_index(drop = True), df_cat], axis=1)
	print(df_patient_admission.shape)

	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs = df_prescriptions_join_filtered.groupby('PAID')['DRUG'].apply(lambda x: x.tolist()).reset_index(name='DRUGS')
	drugs_explode = drugs.explode('DRUGS')
	drugs_explode['occurrence'] = 1
	drugs_new = pd.pivot_table(drugs_explode, index='PAID', columns='DRUGS', values = 'occurrence', fill_value=0)
	drugs_new.reset_index(inplace=True)

	# pca drugs
	drugs_pca = drugs_new.drop('PAID', axis = 1)
	pca = PCA(n_components=40)
	pca.fit(drugs_pca)
	drugs_pca = pca.transform(drugs_pca)
	drugs_pca = pd.DataFrame(drugs_pca)
	drugs_new = pd.concat([drugs_new['PAID'], drugs_pca], axis = 1)
	print(drugs_new.shape)

	# join drugs pca with patient admission
	df_result = df_patient_admission.copy()
	df_result = df_result.drop(['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW'], axis=1)
	target_col = 'mortality'
	data = df_result
	data = move_target_to_end(data, target_col)

	# scale
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	data[num_cols] = scaler.fit_transform(data[num_cols])

	# # pca
	pca = PCA(n_components=20)
	pca.fit(data.drop(target_col, axis=1))
	data_pca = pca.transform(data.drop(target_col, axis=1))

	# final table
	data_pca = pd.DataFrame(data_pca)
	data = pd.concat([df_patient_admission['PAID'], data_pca, data[target_col]], axis = 1)
	data_final = data.merge(drugs_new, on = 'PAID', how = 'inner')
	data = data_final.drop('PAID', axis = 1)
	data = move_target_to_end(data, target_col)
	data.columns = [str(col) for col in data.columns]
	print(data.shape)

	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config


def process_mimiciii_mo2():
	
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')

	###########################################################################################################################
	# patient informatin
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	df_patient_admission.reset_index(drop=True, inplace=True)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())

	df_num = df_patient_admission[['LENGTH_OF_STAY', 'AGE']].copy()
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	df_num[num_cols] = scaler.fit_transform(df_num[num_cols])

	df_patient_info = pd.concat([df_patient_admission['PAID'], df_cat, df_num], axis=1)
	print(df_patient_info.shape)

	patient_info_pca = convert_pca(df_patient_info, 10, prefix = 'pa')

	###############################################################################################################################
	# Drugs
	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs_counts = df_prescriptions_join_filtered.groupby(['PAID', 'DRUG']).size().reset_index(name='occurrence')
	drugs = pd.pivot_table(drugs_counts, index='PAID', columns='DRUG', values = 'occurrence', fill_value=0)
	drugs.reset_index(inplace=True)

	# pca
	drugs = drugs.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	drugs.fillna(0, inplace=True)
	print(drugs.shape)

	drugs_pca = convert_pca(drugs, 20, prefix = 'drug')

	###############################################################################################################################
	# Diagnosis
	df_diagnosis = pd.read_csv('data/mimiciii/DIAGNOSES_ICD.csv')
	df_diagnosis['PAID'] = df_diagnosis['SUBJECT_ID'].astype(str) + '_' + df_diagnosis['HADM_ID'].astype(str)
	df_diagnosis = df_diagnosis[['PAID', 'ICD9_CODE']]
	df_diagnosis = df_diagnosis.dropna()
	df_diagnosis['ICD9_GROUP'] = df_diagnosis['ICD9_CODE'].apply(lambda row: icd_group(row))
	df_diagnosis = df_diagnosis.drop('ICD9_CODE', axis=1)

	diagnosis_counts = df_diagnosis.groupby(['PAID', 'ICD9_GROUP']).size().reset_index(name='occurrence')
	diagnosis = pd.pivot_table(diagnosis_counts, index='PAID', columns='ICD9_GROUP', values = 'occurrence', fill_value=0)
	diagnosis.reset_index(inplace=True)
	diagnosis = diagnosis.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	diagnosis.fillna(0, inplace=True)
	print(diagnosis.shape)

	diagnosis_pca = convert_pca(diagnosis, 10, prefix = 'diag')
	diagnosis_pca

	###############################################################################################################################
	# Procedure
	df_procedure = pd.read_csv('data/mimiciii/PROCEDUREEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	procedure = convert_df(df_procedure)
	procedure = procedure.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	procedure.fillna(0, inplace=True)
	print(procedure.shape)

	procedures_pca = convert_pca(procedure, 10, prefix = 'proc')

	###############################################################################################################################
	# Input events
	df_input = pd.read_csv('data/mimiciii/INPUTEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	inputevents = convert_df(df_input)
	inputevents = inputevents.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	inputevents.fillna(0, inplace=True)
	print(inputevents.shape)

	inputevents_pca = convert_pca(inputevents, 10, prefix = 'input')

	###############################################################################################################################
	# join drugs pca with patient admission
	df_result = patient_info_pca.copy()
	df_result = pd.concat([df_result, df_patient_admission['mortality']], axis = 1)
	df_result = df_result.merge(drugs_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(inputevents_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(diagnosis_pca, on = 'PAID', how = 'left')
	df_result = df_result.merge(procedures_pca, on = 'PAID', how = 'left')
	df_result = df_result.fillna(0)
	print(df_result.shape)

	target_col = 'mortality'
	data = df_result
	data = data.drop('PAID', axis = 1)
	data.columns = [str(col) for col in data.columns]
	data = move_target_to_end(data, target_col)

	data.columns = [str(col) for col in data.columns]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)
	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config


def process_mimic_icd():
	df_diagnosis = pd.read_csv('data/mimiciii/DIAGNOSES_ICD.csv')
	df_diagnosis = df_diagnosis[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'SEQ_NUM']]
	df_diagnosis = df_diagnosis.dropna()
	df_diagnosis['PAID'] = df_diagnosis['SUBJECT_ID'].astype(str) + '_' + df_diagnosis['HADM_ID'].astype(str)
	df_diagnosis['ICD9_GROUP'] = df_diagnosis['ICD9_CODE'].apply(lambda row: icd_group(row))
	merged_cats = [16, 17, 11, 12, 13, 20, 13, 15, 4, 10]
	most_frequent_icd = df_diagnosis.groupby(['PAID'])['ICD9_GROUP'].apply(lambda x: x.mode().iloc[0]).reset_index()
	most_frequent_icd['ICD9_GROUP'] = most_frequent_icd['ICD9_GROUP'].apply(lambda x: x if x not in merged_cats else 21)
	most_frequent_icd['ICD9_GROUP'].value_counts()
	print(most_frequent_icd.shape)

	###########################################################################################################################
	# patient informatin
	df_patient = pd.read_csv('data/mimiciii/PATIENTS.csv').set_index('ROW_ID')
	df_admissions = pd.read_csv('data/mimiciii/ADMISSIONS.csv').set_index('ROW_ID')
	df_patient_admission = df_patient.merge(df_admissions, on='SUBJECT_ID', how='inner')
	df_patient_admission = df_patient_admission.drop([
		'DOD_HOSP', 'DOD_SSN', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'], axis=1)
	df_patient_admission['PAID'] = df_patient_admission['SUBJECT_ID'].astype(str) + '_' + df_patient_admission['HADM_ID'].astype(str)
	df_patient_admission['ADMITTIME'] = pd.to_datetime(df_patient_admission['ADMITTIME'])
	df_patient_admission['DISCHTIME'] = pd.to_datetime(df_patient_admission['DISCHTIME'])
	df_patient_admission['EDREGTIME'] = pd.to_datetime(df_patient_admission['EDREGTIME'])
	df_patient_admission['EDOUTTIME'] = pd.to_datetime(df_patient_admission['EDOUTTIME'])
	df_patient_admission['DOB'] = pd.to_datetime(df_patient_admission['DOB'])
	df_patient_admission['DOD'] = pd.to_datetime(df_patient_admission['DOD'])
	admission_count = df_patient_admission.groupby('SUBJECT_ID')['HADM_ID'].count()
	df_patient_admission['ADMISSION_COUNT'] = df_patient_admission['SUBJECT_ID'].map(admission_count)

	# admission time
	def update_admission_time(row):
		if pd.isna(row['EDREGTIME']):
			return row['ADMITTIME']
		else:
			if (row['ADMITTIME'] - row['EDREGTIME']) > np.timedelta64(0, 's'):
				return row['EDREGTIME']
			else:
				return row['ADMITTIME']

	def update_disch_time(row):
		if pd.isna(row['EDOUTTIME']):
			return row['DISCHTIME']
		else:
			if (row['DISCHTIME'] - row['EDOUTTIME']) > np.timedelta64(0, 's'):
				return row['DISCHTIME']
			else:
				return row['EDOUTTIME']

	df_patient_admission['ADMITTIME_NEW'] = df_patient_admission.apply(lambda x: update_admission_time(x), axis=1)
	df_patient_admission['DISCHTIME_NEW'] = df_patient_admission.apply(lambda x: update_disch_time(x), axis=1)
	length_of_stay = ((df_patient_admission['DISCHTIME_NEW'] - df_patient_admission['ADMITTIME_NEW'])/np.timedelta64(1, 's'))
	df_patient_admission['LENGTH_OF_STAY'] = length_of_stay
	df_patient_admission = df_patient_admission[df_patient_admission['LENGTH_OF_STAY'] > 0]

	# mortality and only consider last time admission
	mortality = (df_patient_admission['DOD'] - df_patient_admission['DISCHTIME_NEW'])/np.timedelta64(1, 'D')
	df_patient_admission['mortality'] = mortality < 365*2
	#df_patient_admission = df_patient_admission.loc[df_patient_admission.groupby('SUBJECT_ID')['ADMITTIME_NEW'].idxmax()].copy()

	# feature engineering
	age = (df_patient_admission['ADMITTIME_NEW'].dt.year - df_patient_admission['DOB'].dt.year)
	age[age >= 90] = 90

	df_patient_admission['AGE'] = age
	df_patient_admission['AGE_IND'] = df_patient_admission['AGE'].apply(lambda x: 1 if x >= 65 else 0)
	df_patient_admission['GENDER'] = df_patient_admission['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
	df_patient_admission.fillna('None', inplace=True)

	cat_cols = [
		'ADMISSION_TYPE', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS',
		'ETHNICITY'
	]

	drop_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'EDREGTIME', 'EDOUTTIME', 'DIAGNOSIS', 'DOB', 'DOD', 'EXPIRE_FLAG']

	df_patient_admission.drop(columns=drop_cols, inplace=True)
	df_patient_admission.reset_index(drop=True, inplace=True)

	# patient information pca
	df_cat = df_patient_admission[cat_cols]
	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(df_cat)
	df_cat = oh.transform(df_cat)
	df_cat = pd.DataFrame(df_cat, columns = oh.get_feature_names_out())

	df_num = df_patient_admission[['LENGTH_OF_STAY', 'AGE']].copy()
	scaler = StandardScaler()
	num_cols = ['LENGTH_OF_STAY', 'AGE']
	df_num[num_cols] = scaler.fit_transform(df_num[num_cols])

	df_patient_info = pd.concat([df_patient_admission['PAID'], df_cat, df_num], axis=1)
	print(df_patient_info.shape)

	patient_info_pca = convert_pca(df_patient_info, 10, prefix = 'pa')

	###############################################################################################################################
	# Drugs
	# fetch drugs 48 hours after admission
	df_pa_date = df_patient_admission[['PAID', 'ADMITTIME_NEW', 'DISCHTIME_NEW']]
	df_prescriptions = pd.read_csv('data/mimiciii/PRESCRIPTIONS.csv', low_memory = False).set_index('ROW_ID')
	df_prescriptions['PAID'] = df_prescriptions['SUBJECT_ID'].astype(str) + '_' + df_prescriptions['HADM_ID'].astype(str)
	df_prescriptions['STARTDATE'] = pd.to_datetime(df_prescriptions['STARTDATE'])
	df_prescriptions['ENDDATE'] = pd.to_datetime(df_prescriptions['ENDDATE'])
	df_prescriptions_join = df_prescriptions.merge(df_pa_date, how='left', on='PAID')
	df_prescriptions_join.dropna(subset=['STARTDATE', 'ENDDATE'], inplace=True)
	time_diff = (df_prescriptions_join['STARTDATE'] - df_prescriptions_join['ADMITTIME_NEW'])/np.timedelta64(1,'D')
	df_prescriptions_join_filtered = df_prescriptions_join[(time_diff <= 2)]

	# get drugs
	drugs_counts = df_prescriptions_join_filtered.groupby(['PAID', 'DRUG']).size().reset_index(name='occurrence')
	drugs = pd.pivot_table(drugs_counts, index='PAID', columns='DRUG', values = 'occurrence', fill_value=0)
	drugs.reset_index(inplace=True)

	# pca
	drugs = drugs.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	drugs.fillna(0, inplace=True)
	print(drugs.shape)

	drugs_pca = convert_pca(drugs, 10, prefix = 'drug')

	###############################################################################################################################
	# Procedure
	df_procedure = pd.read_csv('data/mimiciii/PROCEDUREEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	procedure = convert_df(df_procedure)
	procedure = procedure.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	procedure.fillna(0, inplace=True)
	print(procedure.shape)

	procedures_pca = convert_pca(procedure, 10, prefix = 'proc')

	###############################################################################################################################
	# Input events
	df_input = pd.read_csv('data/mimiciii/INPUTEVENTS_MV.csv', low_memory = False).set_index('ROW_ID')
	inputevents = convert_df(df_input)
	inputevents = inputevents.merge(df_patient_admission[['PAID']], how='right', on='PAID')
	inputevents.fillna(0, inplace=True)
	print(inputevents.shape)

	inputevents_pca = convert_pca(inputevents, 10, prefix = 'input')

	###############################################################################################################################
	# merge all tables
	df_result = most_frequent_icd.copy()
	df_result = df_result.merge(patient_info_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(drugs_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(inputevents_pca, on = 'PAID', how = 'inner')
	df_result = df_result.merge(procedures_pca, on = 'PAID', how = 'inner')
	print(df_result.shape)

	target_col = 'ICD9_GROUP'
	data = df_result
	data = data.drop('PAID', axis = 1)
	data.columns = [str(col) for col in data.columns]
	data = move_target_to_end(data, target_col)
	data[target_col] = pd.factorize(data[target_col])[0]

	data.columns = [str(col) for col in data.columns]
	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)
	data = data.sample(n = 20000, random_state = 42)
	data[target_col] = data[target_col].astype(int)
	data.reset_index(drop=True, inplace=True)

	print(data.shape)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	
	return data, data_config


def process_mimic_mo():

	data = pd.read_csv('data/mimiciii/mimic_mo.csv', low_memory=False)
	target_col = 'mortality'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config

def process_mimic_icd2():

	data = pd.read_csv('data/mimiciii/mimic_icd.csv', low_memory=False)
	target_col ='ICD9_GROUP'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config

def process_mimic_los():

	data = pd.read_csv('data/mimiciii/mimic_los.csv', low_memory=False)
	target_col = 'LENGTH_OF_STAY'
	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'regression',
		'data_type': 'tabular'
	}

	print(data.shape)
	
	return data, data_config


def process_genetic(sample = False):
	data = pd.read_csv('data/genetic/clinvar_conflicting.csv', low_memory=False)
	data.drop(["CLNHGVS"],axis = 1, inplace = True)
	target_col = "CLASS"
	data = move_target_to_end(data,target_col)

	def value_correction(df,columns):
	
		def correction(row):
			if pd.isna(row):
				return row
			else:
				if "/" in row:
					return float(row.split("/")[0])/float(row.split("/")[1])
				else:
					first_value = row.split("-")[0]
					if first_value == "?":
						first_value = row.split("-")[1]
					return first_value

		for col in columns:

			df[col] = df[col].apply(correction)
			df[col] = df[col].astype(float)

		return df

	data = value_correction(data,["CDS_position","cDNA_position","Protein_position", "INTRON","EXON"])
	data.drop(["CDS_position","cDNA_position"],axis = 1, inplace = True)
	data['EXON'] = data["EXON"].fillna(data['INTRON'])
	data.drop(["INTRON"], axis = 1, inplace = True)

	# drop missing columns 
	ms_ratio_df = data.isnull().sum()/data.shape[0]
	high_ms_cols = ms_ratio_df[ms_ratio_df>0.95].index
	print(high_ms_cols)
	data = data.drop(high_ms_cols,axis=1)

	# impute missing values
	ms = data.isnull().sum()/data.shape[0]
	ms = ms[ms>0].sort_values(ascending=False)

	cat_missing_cols = [
	'PolyPhen', 'SIFT', 'CLNVI', 'BAM_EDIT', 'Amino_acids', 'Codons', 'MC', 'SYMBOL', 'Feature', 'Feature_type', 'BIOTYPE'
	]

	num_missing_cols = [ col for col in ms.index if col not in cat_missing_cols]

	assert len(cat_missing_cols) + len(num_missing_cols) == len(ms)

	data[cat_missing_cols] = data[cat_missing_cols].fillna(data[cat_missing_cols].mode().iloc[0])
	data[num_missing_cols] = data[num_missing_cols].fillna(data[num_missing_cols].mean())

	# drop rows with missing values
	# data = data.dropna(axis=0)
	# data = data[data['CHROM'] != 'X']
	# data['CHROM'] = data['CHROM'].astype(int)

	# # categorical columns and numerical columns
	cat_cols = [
		'REF', 'ALT', 'CLNDISDB', 'CLNDN', 'CLNVC', 'MC', 'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
		'Feature', 'BIOTYPE', 'Amino_acids', 'Codons', 'PolyPhen', 'SIFT',  'CLNVI', 'BAM_EDIT', 'CHROM'
	]

	num_cols = [col for col in data.columns if col not in cat_cols and col != target_col]
	# onehot encoding categorical columns
	assert len(cat_cols) + len(num_cols) + 1 == data.shape[1]

	oh = OneHotEncoder(sparse_output=False, drop="if_binary", max_categories=15)
	oh.fit(data[cat_cols])
	oh_cols = oh.get_feature_names_out()
	oh_data = oh.transform(data[cat_cols])
	oh_data = pd.DataFrame(oh_data, columns=oh_cols, index=data.index)

	print(oh_data.shape)

	# standardize numerical columns
	scaler = StandardScaler()
	scaler.fit(data[num_cols])
	num_data = scaler.transform(data[num_cols])
	num_data = pd.DataFrame(num_data, columns=num_cols, index=data.index)
	print(num_data.shape)

	# combine categorical and numerical columns
	data = pd.concat([oh_data, num_data, data[target_col]], axis=1)
	print(data.shape)

	# pca
	pca = PCA(n_components=0.9)
	pca.fit(data.drop(target_col, axis=1))
	pca_data = pca.transform(data.drop(target_col, axis=1))
	pca_data = pd.DataFrame(pca_data, index=data.index)
	data = pd.concat([pca_data, data[target_col]], axis=1)

	data = convert_gaussian(data, target_col)
	data = normalization(data, target_col)

	if sample:
		data0 = data[data[target_col] == 0]
		data1 = data[data[target_col] == 1]
		if data0.shape[0] > data1.shape[0]:
			data0 = data0.sample(n=data1.shape[0], random_state=0)
		else:
			data1 = data1.sample(n=data0.shape[0], random_state=0)

		data = pd.concat([data0, data1], axis=0)

	if data.shape[0] > 20000:
		data = data.sample(n=20000, random_state=0)

	data_config = {
		'target': target_col,
		'important_features_idx': [],
		'features_idx': [idx for idx in range(0, data.shape[1]) if data.columns[idx] != target_col],
		"num_cols": data.shape[1] - 1,
		'task_type': 'classification',
		'clf_type': 'binary-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	return data, data_config


#############################################################################################################
# Utilities
#############################################################################################################
def icd_group(x):
    if isinstance(x, str) and (x[0] == 'V'  or x[0] == 'E'):
        if x[0] == 'V':
            return 19
        elif x[0] == 'E':
            return 20
    else:
        icd_int = int(x[0:3])
        if icd_int < 140:
            return 1
        elif icd_int < 240:
            return 2
        elif icd_int < 280:
            return 3
        elif icd_int < 290:
            return 4
        elif icd_int < 320:
            return 5
        elif icd_int < 390:
            return 6
        elif icd_int < 460:
            return 7
        elif icd_int < 520:
            return 8
        elif icd_int < 580:
            return 9
        elif icd_int < 630:
            return 10
        elif icd_int < 680:
            return 11
        elif icd_int < 710:
            return 12
        elif icd_int < 740:
            return 13
        elif icd_int < 780:
            return 14
        elif icd_int < 790:
            return 15
        elif icd_int < 797:
            return 16
        elif icd_int < 800:
            return 17
        else:
            return 18

def convert_df(df):
    df['PAID'] = df['SUBJECT_ID'].astype(str) + '_' + df['HADM_ID'].astype(str)
    df = df[['PAID', 'ITEMID']]
    counts = df.groupby(['PAID', 'ITEMID']).size().reset_index(name='occurrence')
    df = pd.pivot_table(counts, index='PAID', columns='ITEMID', values='occurrence', fill_value=0)
    df = df.reset_index()
    return df.copy()

def convert_pca(df, n_components=20, prefix='patient'):
    df_pca = df.drop('PAID', axis = 1)
    pca = PCA(n_components=n_components)
    pca.fit(df_pca)
    df_pca = pca.transform(df_pca)
    df_pca = pd.DataFrame(df_pca)
    df_pca.columns = [prefix + str(i) for i in range(df_pca.shape[1])]
    df_pca = pd.concat([df['PAID'], df_pca], axis = 1)
    return df_pca.copy()