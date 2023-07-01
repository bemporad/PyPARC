# -*- coding: utf-8 -*-
"""
Example of PWA regression and classification by block-coordinate descent
on PMLB datasets.

See https://epistasislab.github.io/pmlb/ for selection of datasets.

(C) 2021-2023 A. Bemporad
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pmlb import fetch_data

from pyparc.parc import PARC

np.random.seed(0)  # for reproducibility of results

#####################
# PARC FOR REGRESSION
#####################

K = 10
alpha = 0.01
beta = 1.0e-3
maxiter = 15

dataset = '609_fri_c0_1000_5'
X, Y = fetch_data(dataset, return_X_y=True)

ny = 1
categorical = False

# Detects and possibly one-hot encode categorical features
Nf, nx = X.shape
xCat = np.zeros(nx, dtype=int)
for i in range(nx):
    xCat[i] = np.unique(X[:, i]).shape[0]
isCat = (xCat <= 4)
if np.any(isCat):
    # one-hot encode categorical features
    drop_enc = preprocessing.OneHotEncoder(drop='first', sparse=False).fit(X[:, isCat])
    Xe = drop_enc.transform(X[:, isCat])
    X = np.hstack((X[:, ~isCat], Xe))
    nx = X.shape[1]
nxc = np.sum(~isCat)

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
if nxc > 0:
    scaler_x = preprocessing.StandardScaler().fit(X_train[:, 0:nxc])
    X_train[:, 0:nxc] = scaler_x.transform(X_train[:, 0:nxc])
    X_test[:, 0:nxc] = scaler_x.transform(X_test[:, 0:nxc])
scaler_y = preprocessing.StandardScaler().fit(Y_train.reshape(-1, 1))
Y_train = scaler_y.transform(Y_train.reshape(-1, 1))
Y_test = scaler_y.transform(Y_test.reshape(-1, 1))

N = X_train.shape[0]

predictor = PARC(K=K, alpha=alpha, maxiter=maxiter, min_number=int(len(Y_train) / 100),
                 beta=beta)
predictor.fit(X_train, Y_train, categorical, weights=np.ones(ny))

# Y_hat, delta_hat = predictor.predict(X_test) # predict targets

score_train = predictor.score(X_train, Y_train)  # score on training data
score_test = predictor.score(X_test, Y_test)  # score on test data

print("\nRegression results (R2 score) on dataset '%s':\n" % dataset)
print("Training data: %6.2f %%" % (score_train[0] * 100))
print("Test data:     %6.2f %%" % (score_test[0] * 100))
print("--------------------\n")

# Create partition only in space of features 1,2
feature_selection=np.zeros(nx, dtype=bool)
feature_selection[0:2]=True
predictor.fit(X_train, Y_train, categorical, weights=np.ones(ny), feature_selection=feature_selection)

score_train_subset = predictor.score(X_train, Y_train)  # score on training data
score_test_subset = predictor.score(X_test, Y_test)  # score on test data

print("\nRegression results (R2 score) on dataset (reduced partition) '%s':\n" % dataset)
print("Training data: %6.2f %%" % (score_train_subset[0] * 100))
print("Test data:     %6.2f %%" % (score_test_subset[0] * 100))
print("--------------------\n")

#####################
# PARC FOR CLASSIFICATION
#####################

K = 5
alpha = 0.02
beta = 1.0e-3
maxiter = 15

dataset = 'credit_g'
X, Y = fetch_data(dataset, return_X_y=True)

ny = 1
categorical = True

# Detects and possibly one-hot encode categorical features
Nf, nx = X.shape
xCat = np.zeros(nx, dtype=int)
for i in range(nx):
    xCat[i] = np.unique(X[:, i]).shape[0]
isCat = (xCat <= 4)
if np.any(isCat):
    # one-hot encode categorical features
    drop_enc = preprocessing.OneHotEncoder(drop='first', sparse=False).fit(X[:, isCat])
    Xe = drop_enc.transform(X[:, isCat])
    X = np.hstack((X[:, ~isCat], Xe))
    nx = X.shape[1]
nxc = np.sum(~isCat)

test_size = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
if nxc > 0:
    scaler_x = preprocessing.StandardScaler().fit(X_train[:, 0:nxc])
    X_train[:, 0:nxc] = scaler_x.transform(X_train[:, 0:nxc])
    X_test[:, 0:nxc] = scaler_x.transform(X_test[:, 0:nxc])

N = X_train.shape[0]

predictor = PARC(K=K, alpha=alpha, maxiter=maxiter, min_number=int(len(Y_train) / 100),
                 beta=beta)
predictor.fit(X_train, Y_train, categorical, weights=np.ones(ny))

# Y_hat, delta_hat = predictor.predict(X_test) # predict targets

score_train = predictor.score(X_train, Y_train)  # score on training data
score_test = predictor.score(X_test, Y_test)  # score on test data

print("\nClassification results (accuracy) on dataset '%s':\n" % dataset)
print("Training data: %6.2f %%" % (score_train[0] * 100))
print("Test data:     %6.2f %%" % (score_test[0] * 100))
print("--------------------\n")
