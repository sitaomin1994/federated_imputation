from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


########################################################################################################################
# Classification
########################################################################################################################
def eval_LR(X_train, y_train, X_test, y_test, seed=21):
	model = LogisticRegression(random_state=seed, solver='liblinear', multi_class='ovr', max_iter=1000)
	grid = {'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}
	clf = GridSearchCV(model, grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return {
		'accuracy': accuracy_score(y_test, y_pred),
		'roc_auc': roc_auc_score(y_test, y_score, multi_class='ovr')
	}


def eval_svm(X_train, y_train, X_test, y_test, seed=21):
	model = SVC(random_state=seed, kernel='rbf', probability=True)
	grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
	clf = GridSearchCV(model, grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return {
		'accuracy': accuracy_score(y_test, y_pred),
		'roc_auc': roc_auc_score(y_test, y_score, multi_class='ovr')
	}


def eval_rf_clf(X_train, y_train, X_test, y_test, seed=21):
	model = RandomForestClassifier(random_state=seed, n_jobs=-1)
	grid = {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [2, 3, 5, 10, 20]}
	clf = GridSearchCV(model, grid, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return {
		'accuracy': accuracy_score(y_test, y_pred)
	}


########################################################################################################################
# Regression
########################################################################################################################
def eval_ridge(X_train, y_train, X_test, y_test, seed = 21):
	model = Ridge(random_state=seed)
	grid = {'alpha': [0.1, 1, 10, 100, 1000]}
	clf = GridSearchCV(model, grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return {
		'rmse': mean_squared_error(y_test, y_pred, squared=False),
		'r2': r2_score(y_test, y_pred),
	}


def eval_svr(X_train, y_train, X_test, y_test, seed = 21):
	model = SVR(kernel='rbf')
	grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]}
	clf = GridSearchCV(model, grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return {
		'rmse': mean_squared_error(y_test, y_pred, squared=False),
		'r2': r2_score(y_test, y_pred),
	}
