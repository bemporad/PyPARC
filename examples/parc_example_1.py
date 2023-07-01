# -*- coding: utf-8 -*-
"""
Example of PWA regression to fit a nonlinear function.

(C) 2021-2023 A. Bemporad
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split

from pyparc.parc import PARC

np.random.seed(0)  # for reproducibility
savefigs=False # save figures as PNG images

# Generate dataset
N = 1000
test_size = 0.2

nx = 2
xmin = 0.
xmax = 1.
ymin = 0.
ymax = 1.
X = np.random.rand(N, nx) * np.array([xmax - xmin, ymax - ymin]) + np.array([xmin, ymin])

nyc = 1  # number of numeric outputs
noise_frac = 0.0  # noise standard deviation (0 = no noise)

f = lambda x1, x2: np.sin(4 * x1 - 5 * (x2 - 0.5) ** 2) + 2 * x2
Y = f(X[:, 0], X[:, 1]) + noise_frac * np.random.randn(N)

# Plot level sets of nonlinear function
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()

dx = (xmax - xmin) / 100.0
dy = (ymax - ymin) / 100.0
[x1, x2] = np.meshgrid(np.arange(xmin, xmax + dx, dx), np.arange(ymin, ymax + dy, dy))
z = f(x1, x2)
nlevels = 8
plt.contourf(x1, x2, z, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, z, linewidths=3.0, levels=nlevels)
plt.title('level sets of y(x)', fontsize=20)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

if savefigs:
    plt.savefig('fig1.png')
else:
    plt.show()

# Get random split of training/test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

# Setup PARC regression algorithm
K = 10          # maximum number of partitions
alpha = 1.0e-4  # L2-regularization
maxiter = 15    # maximum number of block-coordinate descent iterations

predictor = PARC(K=K, alpha=alpha, maxiter=maxiter)

# Y_hat, delta_hat = predictor.predict(X_test) # predict targets

# Solve PWA regression problem
categorical = False # we have a numeric target
predictor.fit(X_train, Y_train)

# Compute R2 scores
score_train = predictor.score(X_train, Y_train)
score_test = predictor.score(X_test, Y_test)

print("\nR2 scores:\n")
print("Training data: %6.2f %%" % (score_train[0] * 100))
print("Test data:     %6.2f %%" % (score_test[0] * 100))
print("--------------------\n")

Kf = predictor.K  # final number of partitions
delta = predictor.delta  # final assignment of training points to clusters
xbar = predictor.xbar  # centroids of final clusters

# Plot level sets of PWA prediction function
zpwl, _ = predictor.predict(np.hstack((x1.reshape(x1.size, 1), x2.reshape(x2.size, 1))))
zpwl = zpwl.reshape(x1.shape)

fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.grid()
ax.set_xlim([xmin, xmax])
ax.set_ylim([0, 1])
NN = x1.shape[0]
plt.contourf(x1, x2, zpwl, alpha=0.6, levels=nlevels)
plt.contour(x1, x2, zpwl, linewidths=3.0, levels=nlevels)
plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig2.png')
else:
    plt.show()

Yhtrain, _ = predictor.predict(X_train)
Yhtest, delta_test = predictor.predict(X_test)

# Plot PWA partition
fig, ax = plt.subplots(figsize=(8, 8))
for i in range(0, Kf):
    iD = (delta == i).ravel()
    plt.scatter(X_train[iD, 0], X_train[iD, 1], marker='*', linewidth=3,
                alpha=0.5, color=cm.tab10(i))
plt.grid()
plt.scatter(xbar[:, 0], xbar[:, 1], marker='o', linewidth=5, alpha=.5, color=(.8, .4, .4))

predictor.plot_partition([xmin, ymin], [xmax, ymax], fontsize=32,
                         alpha=.4, linestyle='-', linewidth=2.0,
                         edgecolor=(0, 0, 0), facecolor=(1,1,1))
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.title('PARC (K = %d)' % K, fontsize=20)

if savefigs:
    plt.savefig('fig3.png')
else:
    plt.show()

# Plot nonlinear function
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
ax.scatter(X_test[:, 0], X_test[:, 1], Y_test, alpha=0.5)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.plot_surface(x1, x2, z, alpha=0.5)
ax.view_init(35, -120)
plt.title('Nonlinear function', fontsize=20)

if savefigs:
    plt.savefig('fig4.png')
else:
    plt.show()

# Plot PWA function
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(r'$x_1$', labelpad=15, fontsize=20)
ax.set_ylabel(r'$x_2$', labelpad=15, fontsize=20)
ax.set_zlabel(r'$y$', fontsize=20)
for i in range(0, Kf):
    iD = (delta_test == i).ravel()
    ax.scatter(X_test[iD, 0], X_test[iD, 1], Y_test[iD], marker='*',
               linewidth=3, alpha=0.5, color=cm.tab10(i))

ax.plot_surface(x1, x2, zpwl, alpha=0.5)
ax.view_init(35, -120)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.title('PARC (K = %d)' % K, fontsize=20)

##################################################################################
# MIP Optimization: find feature vector 'x' such that the corresponding target 'y'
# is as close as possible to a given reference value 'yref'
##################################################################################
solveMIP = False
if solveMIP:
    Xmin = np.min(X, axis=0)
    Xmax = np.max(X, axis=0)
    yref = 3. * np.ones(nyc)  # desired target
    verbose = False
    #solver = "CBC" # Use CBC solver
    solver="GRB" # Use Gurobi's MILP solver
    x, y, region, f = predictor.optimize(Xmin, Xmax, yref, verbose=verbose, solver=solver)
    ax.scatter(x[0], x[1], y, marker='o', linewidth=10.0, color=(1, 0, 0))
    print("yhat = %5.4f, x1 = %5.4f, x2 = %5.4f" % (y, x[0], x[1]))

if savefigs:
    plt.savefig('fig5.png')
else:
    plt.show()

##################################################################################
# Choose best number of partitions by K-fold cross validation
##################################################################################
runCrossValidation = False
if runCrossValidation:
    Ks = [5, 10, 15] # values of K tested
    bestK, results = predictor.cross_val(X_train, Y_train, categorical, Kfolds=5, Ks=Ks)