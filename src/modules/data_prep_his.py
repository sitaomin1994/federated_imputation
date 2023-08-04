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
		'clf_type': 'multi-class',
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
		'clf_type': 'multi-class',
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
	data = data.drop_duplicates(subset= ['patient_nbr'], keep = 'first')

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
		sparse_output=False, handle_unknown='ignore', max_categories=10, drop='first'
	)

	X_cat = oh_encoder.fit_transform(data[cat_cols].values)

	# min-max scaling
	scaler = StandardScaler()
	X_num = scaler.fit_transform(data[num_cols].values)

	# combine
	X = np.concatenate([X_num, X_cat], axis=1)
	y = data[target_col].values

	# pca
	pca = PCA(n_components=0.9)
	X = pca.fit_transform(X)

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
		'clf_type': 'multi-class',
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
		'clf_type': 'multi-class',
		'data_type': 'tabular'
	}

	print(data.shape)
	print(data[target_col].value_counts())
	
	if verbose:
		logger.debug("Important features {}".format(important_features))
		logger.debug("Data shape {}".format(data.shape, data.shape))
		logger.debug(data_config)
	
	return data, data_config