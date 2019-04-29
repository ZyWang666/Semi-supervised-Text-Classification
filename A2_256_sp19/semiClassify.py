#!/bin/python

def train_classifier(X, y,ws):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import GridSearchCV
	param_grid = {'C': [0.001, 0.01, 0.1, 1,1.1,1.2,1.3,1.4,1.5, 2,3,4,5, 6,7,8,9,10]}
	grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
	grid.fit(X, y)
	#print("Best cross-validation score: {:.2f}".format(grid.best_score_))
	#print("Best parameters: ", grid.best_params_)
	#print("Best estimator: ", grid.best_estimator_)
	cls = grid.best_estimator_
	#cls = LogisticRegression(C=1,random_state=0, solver='lbfgs', max_iter=10000, warm_start=ws, fit_intercept=True)
	cls.fit(X, y)
	return cls


def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))
