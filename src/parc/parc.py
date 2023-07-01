"""
PARC - Piecewise Affine Regression and Classification

(C) 2021-2023 A. Bemporad
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from scipy.special import logsumexp

from sklearn.model_selection import KFold  # this is only required by the "cross_val" method
from mip import *  # this is only required by the "optimize" method
import pypoman  # this is only required by the "plot_partition" method
import matplotlib.pyplot as plt  # this is only required by the "plot_partition" method
from matplotlib.patches import Polygon  # this is only required by the "plot_partition" method
from matplotlib.collections import PatchCollection
import faiss  # this is only required by "parc_init" method for k-nearest neighbors finding
import pickle # this is only required for load and save operations


class PARC:
    """PARC - Piecewise Affine Regression and Classification
    """

    def __init__(self, K=10, alpha=1.0e-5, maxiter=100, sigma=1., separation='Softmax',
                 cost_tol=1e-4, min_number=None,
                 fit_on_partition=True, softmax_solver='lbfgs',
                 softmax_maxiter=100000, beta=1.e-3, initialization=("kmeans", 10),
                 verbose=1):

        """PARC - Piecewise Affine Regression and Classification.
        (C) 2021-2023 A. Bemporad

        Parameters:
        ----------
        K : int
            number of linear affine regressor/classifiers in PWA predictor.

        alpha : float
            L2-regularization term.

        maxiter : int
            maximum number of block-coordinate descent iterations.

        sigma : float
            tradeoff coefficient between PWL separability and quality of target fit.

        separation : str
            type of PWL separation used, either 'Voronoi' or 'Softmax'.

        cost_tol : float
            tolerance used to stop block-coordinate descent.

        min_number : int
            minimum number of points allowed per cluster. At the end
            of the procedure, points in excessively small clusters
            are reassigned to cluster of closest point (default: nx+1).

        fit_on_partition : bool
            if True, recompute the best linear model in each polyhedron
            defined by the final PWL partion, otherwise keep the best linear
            models determined based on clusters. If True, also reassign
            points based on the polyhedron they belong to.

        softmax_solver :  str
            solver used in softmax regression.

        softmax_maxiter :  int
            max number of iterations in softmax regression.

        beta : float
            L2-regularization used in softmax regression for PWL separation
            (only used when separation = 'Softmax')

        initialization : (str, int)
            type of initial clustering. Valid options:

            ("kmeans",n_init): uses k-means++ in feature space to get initial clustering,
                          with n_init = number of time the k-means algorithm is run with different centroid seeds
                          (default). Only features such that feature_selection[i]=True are taken into account.

            ("random",0): use random initial assignment.

            ("regression", kn): for each datapoint, performs linear regression on kn nearest neighbors,
                          then performs k-means on the vectors of regression coefficients. For example,
                          kn = 2*(nx + 1), where nx = number of features
                          Only supported for regression of scalar targets.

        verbose : int
            verbosity level (0 = none).

        """

        self.K = K
        self.maxiter = maxiter
        self.alpha = alpha
        self.separation = separation
        self.beta = beta
        self.verbose = verbose
        self.initialization = initialization
        self.sigma = sigma
        self.cost_tol = cost_tol
        self.min_number = min_number
        self.fit_on_partition = fit_on_partition
        self.softmax_solver = softmax_solver
        self.softmax_maxiter = softmax_maxiter

    def fit_PWL(self, softmax_reg, X, delta):
        try:
            softmax_reg.fit(X, delta)
        except:
            softmax_reg.warm_start = False  # new problem has different number of classes, disable warm start
            softmax_reg.fit(X, delta)
            softmax_reg.warm_start = True
        if len(np.unique(delta)) > 2:
            omega1 = softmax_reg.coef_
            gamma1 = softmax_reg.intercept_
        else:
            omega1 = np.zeros((2, X.shape[1]))
            gamma1 = np.zeros(2)
            omega1[0, :] = -softmax_reg.coef_
            gamma1[0] = -softmax_reg.intercept_
            omega1[1, :] = softmax_reg.coef_
            gamma1[1] = softmax_reg.intercept_

        return omega1, gamma1

    def fit(self, X, Y, categorical=None, weights=None, feature_selection=None):
        '''
        PARC - Fit Piecewise Affine Regression and Classification.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        X : int ndarray of shape (number of data, number of features).
            We assume that categorical features have already been one-hot encoded as 0/1 features.

        Y : int ndarray of shape (number of data, number of targets).
            We assume for each categorical target its categories are encoded as distinct numbers.
            For example, as {0,1} in case of binary targets and {1,2,3,4} for in case of
            4 possible categories.

        categorical : bool ndarray of shape (number of targets).
            Specifies whether a target is numeric (False) or categorical (True).
            For example, if y[0],y[1] are numeric (real-valued), and y[2],y[3] are
            categorical set categorical=[False,False,True,True].
            Default: all targets are numeric.

        weights : float ndarray of shape(number of targets)
            weights on output components. Default = 1.

        feature_selection : bool ndarray of shape (number of features).
            Specifies whether a feature should contibute to define the PWA partition or not.
            The partition is defined in the space of features x[i] for which
            feature_selection[i] = True. Note that all features are instead taken into
            account for linear regression and/or classification in each partition.

        Returns:
        --------
        self.coef_ :   float ndarray of shape (number of clusters, number of features, number of linear target functions).
                       Coefficients associated with linear prediction models.
        self.intercept_ : float ndarray of shape (number of clusters, number of linear target functions).
                       Intercepts associated with linear prediction models.
        self.delta :   int ndarray of shape (number of training data).
                       Assignment of training datapoints to cluster. Points belonging to
                       eliminated small clusters are considered as outliers and
                       marked with delta[k]=-1.
        self.K :       int.
                       Final number of clusters.
        self.iters :   int.
                       Number of block-coordinate descent iterations,
        self.cost :    float ndarray of shape (number of iterations).
                       Sequence of cost function values encountered during block-coordinate descent.
        self.omega :   float ndarray of shape (number of clusters, number of features).
                       Coefficients of final PWL separation function. Column #i of omega is zero
                       when feature_selection[i]=False.
        self.gamma :   float ndarray of shape (number of clusters).
                       Intercepts of final PWL separation function.
        self.xbar :    float ndarray of shape (number of clusters, number of features).
                       Centroids of final clusters.
        self.Nk :      int ndarray of shape (number of clusters).
                       Number of datapoints in each cluster at termination.
        '''

        def fit_coeff(ii):
            # Given current cluster of points indexed by vector ii,
            # compute ridge regression/softmax regression problems, one per target

            if not np.all(categorical):
                # Initialize ridge regressor
                ridge = Ridge(alpha=alpha, fit_intercept=True)

            h = 0
            for i in range(ny):
                if not categorical[i]:
                    ridge.fit(X[ii, :], Yt[ii, i])
                    a[j][:, h] = ridge.coef_
                    b[j, h] = ridge.intercept_
                    h += 1
                else:
                    softmax_reg = softmax_regs[j][i]

                    tot_elems = cat_values[i]  # categories in entire dataset
                    elems = np.unique(Yt[ii, i])  # categories in this cluster (ordered)
                    n_elems = len(elems)
                    if n_elems < numcat[i]:
                        # Possibly missing category values in this cluster still require their
                        # corresponding a,b coefficients/intercepts to be optimized.
                        # Therefore, we introduce here fake data points whose values
                        # equals the missing values (so to maintain coef/intercept order)
                        # and with zero weight
                        dn = numcat[i] - n_elems
                        softmax_weights = np.ones(Nk[j] + dn)
                        softmax_weights[0:dn] = 0.0
                        fake_values = np.setdiff1d(tot_elems, elems, assume_unique=True)

                        softmax_reg.fit(np.vstack((np.zeros((dn, nx)), X[ii, :])),
                                        np.vstack((fake_values.reshape(-1, 1),
                                                   Yt[ii, i].reshape(Nk[j], 1))).ravel(),
                                        sample_weight=softmax_weights)
                    else:
                        # no category is missing
                        softmax_reg.fit(X[ii, :], Yt[ii, i].ravel())

                    if numcat[i] == 2:
                        # binary target

                        # In this case LogisticRegression only returns one coeff_ and intercept_ value.
                        # LogisticRegression associates +coeff_/+intercept_ with **second** category (True),
                        # -coeff_/-intercept_ with **first** category (False). As category numbers are
                        # ordered from smallest to largest, the smallest value corresponds to False.

                        a[j][:, h] = -softmax_reg.coef_
                        b[j, h] = -softmax_reg.intercept_
                        h += 1
                        a[j][:, h] = softmax_reg.coef_
                        b[j, h] = softmax_reg.intercept_
                        h += 1

                        ##########
                        # DEBUG
                        ##########
                        # Y_pred = softmax_reg.predict(X[ii, :])
                        # from sklearn.metrics import accuracy_score
                        # print(accuracy_score(Y[ii, i], Y_pred))
                        ##########

                    else:
                        # multi-category softmax, each category has its own coeff_/intercept_
                        for t in range(numcat[i]):
                            a[j][:, h] = softmax_reg.coef_[t, :]
                            b[j, h] = softmax_reg.intercept_[t]
                            h += 1  # update coefficient/intercept index
            return

        def parc_init():
            # Get initial clustering

            if K == 1:
                return np.zeros(N, dtype=int)

            raiseerror = False
            if not isinstance(self.initialization, tuple):
                raiseerror = True
            if (not raiseerror):
                if len(self.initialization) != 2:
                    raiseerror = True
            if raiseerror:
                raise Exception('initialization option must be a tuple (string, value).')

            init_type = self.initialization[0]

            if init_type == "regression":
                if (ny > 1 or categorical[0]):
                    raise Exception('initialization option "regression" only applicable to scalar numeric target.')

                # Get initialization by K-means on coefficients
                # obtained from linear regression on k_neigh nearest neighbors.
                # Sometimes this provides a very good initial guess.

                delta = np.zeros(N, dtype=int)
                k_neigh = self.initialization[1]  # For example: 2*(nx + 1)
                index = faiss.IndexFlatL2(nx)
                index.add(X.astype(np.float32))

                _, ind = index.search(X.astype(np.float32),
                                      k=k_neigh + 1)  # this return distances and indices of K-nearest neighbors

                # Perform K-means in coefficient-space
                a = np.zeros((N, nx + 1))
                for k in range(N):
                    # Fit hyperplane
                    # ab = arg min ||[X,1]*[a;b]-Y||_2**2
                    zstar = np.linalg.lstsq(np.hstack((X[ind[k, 1:], :], np.ones((k_neigh, 1)))),
                                            Yt[ind[k, 1:], i], rcond=None)[0]
                    a[k, 0:nx] = zstar[0:nx]
                    a[k, nx] = zstar[nx]

                kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10).fit(a)
                delta = kmeans.labels_

            elif init_type == "kmeans":
                # Perform assignment based on feature vectors only such that feature_selection = True
                # Run K-means++ to initialize delta
                kmeans = KMeans(n_clusters=K, init='k-means++',
                                n_init=self.initialization[1]).fit(X[:, feature_selection])
                delta = kmeans.labels_

            elif init_type == 'random':
                delta = np.random.random_integers(0, K - 1, N)  # random assignment
            else:
                raise Exception('Unknown initialization option "%s".' % init_type)

            return delta

        # Initialization
        N, nx = X.shape

        Yt = Y
        throwerror = False
        if len(Yt.shape) == 1:
            Ny = len(Yt)
            if Ny == N:
                Yt = Yt.reshape(N, 1)
            else:
                throwerror = True
        else:
            Ny = Yt.shape[0]
            if not Ny == N:
                throwerror = True
        if throwerror:
            raise Exception('Number of feature vectors and targets is not consistent.')
        ny = Yt.shape[1]

        if categorical is None:
            categorical = np.zeros(ny)

        if np.isscalar(categorical):
            categorical = [categorical]

        categorical=np.array(categorical, dtype=bool)  # convert to np.array and Boolean type

        if len(categorical) != ny:
            raise Exception('input argument "categorical" must have %d Boolean entries, one per output' % ny)

        if weights is None:
            weights = np.ones(ny)
        if self.min_number is None:
            min_number = nx + 1
        else:
            min_number = self.min_number

        if feature_selection is None:
            feature_selection = np.ones(nx, dtype=bool)
        elif not np.any(feature_selection):
            self.K = 1  # no feature to partition, set K=1
        else:
            feature_selection = feature_selection.astype(bool)

        K = int(self.K)
        alpha = float(self.alpha)
        maxiter = int(self.maxiter)
        cost_tol = float(self.cost_tol)
        fit_on_partition = self.fit_on_partition
        separation = self.separation
        verbose = self.verbose

        numcat = np.zeros(ny, dtype=int)  # number of categories (only useful for categorical targets)
        cat_values = []  # store categories here (empty if target is numeric)
        for i in range(ny):
            if categorical[i]:
                yvalues = np.unique(Yt[:, i])
                numcat[i] = len(yvalues)
                cat_values.append(yvalues)
            else:
                cat_values.append([])
        nlin = np.sum(~np.array(categorical)) + np.sum(numcat)  # total number of coefficient/intercept pairs

        # Store indices in coefficient/intercept array corresponding to each output
        coeff_index = []
        h = 0
        for i in range(ny):
            if not categorical[i]:
                coeff_index.append(h)
                h += 1
            else:
                coeff_index.append(np.arange(h, h + numcat[i]).tolist())
                h += h + numcat[i]

        a = np.zeros((K, nx, nlin))  # solution coefficients
        b = np.zeros((K, nlin))  # solution intercepts

        sigma = self.sigma / N

        ##############
        # Initial assignment
        ##############
        delta = parc_init()

        isVoronoi = (separation == 'Voronoi')
        isSoftmax = (separation == 'Softmax')

        xbar = np.zeros((K, nx))
        if isSoftmax:
            PWLsoftmax = LogisticRegression(multi_class='multinomial', C=2.0 / self.beta,
                                            solver=self.softmax_solver,
                                            max_iter=self.softmax_maxiter, tol=1e-6,
                                            warm_start=True)
        omega = np.zeros((K, nx))
        gamma = np.zeros(K)

        Nk = np.zeros(K, dtype=int)  # number of points in cluster
        killed = np.zeros(K, dtype=bool)  # clusters that have disappeared

        # Prepare softmax regression functions, one per categorical output and per cluster
        # so to better exploit warm-start.
        softmax_regs = [[None] * ny] * K
        for j in range(K):
            for i in range(ny):
                if categorical[i]:
                    softmax_regs[j][i] = LogisticRegression(C=0.5 / alpha, tol=1e-4, fit_intercept=True,
                                                            solver=self.softmax_solver,
                                                            max_iter=self.softmax_maxiter, class_weight=None,
                                                            multi_class='multinomial', warm_start=True)
        go = True
        iters = 0
        cost_old = np.inf
        cost_sequence = []
        while go:
            iters += 1

            ##########
            # Solve K ridge regression problems for the numeric targets
            # or ridge classifier for 0/1 targets. Also computes current centroids
            ##########

            for j in range(K):
                if not killed[j]:
                    # Check if some clusters are empty
                    ii = (delta == j)
                    Nk[j] = np.sum(ii)
                    if Nk[j] == 0:
                        killed[j] = True
                    else:
                        if isVoronoi:
                            xbar[j, feature_selection] = np.sum(X[ii].T[feature_selection].T, axis=0) / Nk[j]

                if not killed[j]:
                    fit_coeff(ii)

            ##########
            # find PWL separation function by softmax regression
            ##########
            if K > 1 and isSoftmax:
                omega1, gamma1 = self.fit_PWL(PWLsoftmax, X[:, feature_selection], delta)  # dim(gamma1) = unique(delta)
                h = 0
                for i in range(nx):
                    if feature_selection[i]:
                        omega[~killed, i] = omega1[:, h]
                        h += 1
                gamma[~killed] = gamma1

            ##########
            # Compute cost and assign labels
            ##########
            cost = 0.0
            for k in range(N):
                cost_k = np.zeros(K)
                x_k = X[k, :].ravel()
                if isSoftmax:
                    lin_terms = np.zeros(K)
                for j in range(K):
                    if not killed[j]:
                        # Get Vy function
                        for i in range(ny):
                            if not categorical[i]:
                                z = np.sum(a[j][:, i] * x_k) + b[j, i]
                                cost_k[j] += weights[i] * (Yt[k, i] - z) ** 2
                            else:
                                lin_terms_y = np.zeros(numcat[i])
                                for t in range(numcat[i]):
                                    i1 = coeff_index[i][t]
                                    lin_terms_y[t] = np.sum(a[j][:, i1] * x_k) + b[j, i1]
                                h = (cat_values[i] == Yt[k, i])
                                cost_k[j] += weights[i] * (logsumexp(lin_terms_y) - lin_terms_y[h])

                        # Get Vx function
                        if isVoronoi:
                            cost_k[j] += sigma * np.sum(
                                (x_k[feature_selection] - xbar[j, feature_selection].ravel()) ** 2)
                        elif isSoftmax:
                            aux = np.sum(omega[j, feature_selection] * x_k[feature_selection]) + gamma[j]
                            cost_k[j] -= sigma * aux
                            lin_terms[j] = aux
                    else:
                        cost_k[j] = np.inf  # this prevents reassignement to killed cluster
                if isSoftmax:
                    cost_k += sigma * logsumexp(lin_terms)

                # reassign labels
                delta[k] = np.argmin(cost_k)

                # compute current cost
                cost += cost_k[delta[k]]

            if verbose:
                print("iter = %3d, cost = %5.4f" % (iters, cost))

            cost_sequence.append(cost)

            if (cost_old - cost <= cost_tol) or (iters == maxiter) or (K == 1):
                go = False

            cost_old = cost

        # update Nk and compute final number of clusters, possibly eliminating
        # empty or very small clusters, then clean-up solution
        for j in range(K):
            Nk[j] = np.sum(delta == j)
        killed = (Nk < min_number)

        isoutside = np.zeros(N, dtype=bool)
        anyoutside = False
        for i in range(K):
            if killed[i]:
                anyoutside = True
                isoutside[delta == i] = True

        NC = K - np.sum(killed)  # final number of clusters
        a = a[~killed][:, :]  # only keeps coefficients of nonempty clusters
        b = b[~killed, :]
        delta[isoutside] = -1  # mark points in small clusters with delta = -1
        Nk = np.zeros(NC, dtype=int)

        # recompute PWL partition based on final assignment
        for j in range(NC):
            ii = (delta == j)  # outliers are automatically excluded as j=0,...,NC-1
            Nk[j] = np.sum(ii)
            if Nk[j] > 0:
                xbar[j, feature_selection] = np.sum(X[ii].T[feature_selection].T, axis=0) / Nk[
                    j]  # compute centroids even if isSoftmax=True
        xbar = xbar[~killed, :]

        if NC > 1:
            if K > 1 and isSoftmax:
                omega = np.zeros((NC, nx))
                omega[:, feature_selection], gamma = self.fit_PWL(PWLsoftmax, X[~isoutside].T[feature_selection].T,
                                                                  delta[
                                                                      ~isoutside])  # dim(gamma1) = unique(delta[~isoutside])
        else:
            omega = np.zeros((1, nx))
            gamma = np.zeros(1)

        if isVoronoi:
            # define omega and gamma from centroids
            omega = xbar.copy()
            gamma = -0.5 * np.sum(omega[:, feature_selection] ** 2, axis=1)

        if not fit_on_partition:
            # re-label existing delta in (0,NC)
            elems = list(set(delta))
            if anyoutside:
                elems.remove(-1)
            for i in range(NC):
                delta[delta == elems[i]] = i
        else:
            # reassign points based on the polyhedron they belong to
            delta = np.argmax(X[:, feature_selection] @ omega[:, feature_selection].T + gamma, axis=1)
            delta[isoutside] = -1

            # some clusters may have disappeared after reassignment
            h = 0
            keep = np.ones(NC, dtype=bool)
            for j in range(NC):
                ii = (delta == j)
                aux = np.sum(ii)
                if aux >= min_number:
                    Nk[j] = aux
                    delta[ii] = h  # relabel points
                    xbar[j, feature_selection] = np.sum(X[ii].T[feature_selection].T, axis=0) / Nk[
                        j]  # recompute centroids even if isSoftmax=True
                    h += 1
                else:
                    delta[ii] = -1  # also mark these as outliers, as they form an excessively small cluster
                    keep[j] = False

            if isVoronoi:
                # define omega and gamma from centroids
                omega = xbar.copy()
                gamma = -0.5 * np.sum(xbar[:, feature_selection] ** 2, axis=1)

            if NC > h:
                omega = omega[keep, :]
                gamma = gamma[keep]
                xbar = xbar[keep, :]
                Nk = Nk[keep]
            NC = h

            # recompute coefficients, based on PWL separation instead of clusters
            a = np.zeros((NC, nx, nlin))
            b = np.zeros((NC, nlin))
            for j in range(NC):
                ii = (delta == j)
                if np.any(ii):
                    fit_coeff(ii)

        self.K = int(NC)
        self.coef_ = a
        self.intercept_ = b
        self.delta = delta
        self.cost = cost_sequence
        self.xbar = xbar
        self.omega = omega
        self.gamma = gamma
        self.Nk = Nk
        self.nx = int(nx)
        self.ny = int(ny)
        self.categorical = categorical
        self.cat_values = cat_values
        self.coeff_index = coeff_index
        self.feature_selection = feature_selection
        self.iters = int(iters)

        return self

    ############################################
    def predict(self, x):
        '''
        PARC - Predictor function.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        X : float ndarray of shape (number of data, number of features).
            Feature vectors. We assume that categorical features have already been one-hot encoded as 0/1 features.

        Returns:
        --------
        Y : float ndarray of shape (number of data, number of targets).
            Predicted targets.

        delta : int ndarray of shape (number of data).
            Cluster assignments.
        '''
        nx = self.nx
        N = x.shape[0]
        issingle = (N == nx)
        if issingle:
            X = x.reshape(1, nx)
            N = 1
        else:
            X = x  # 2D array, each row is a different values of x

        ny = self.ny
        yhat = np.zeros((N, ny))

        # find where x is in PWL partition
        delta = np.argmax(X[:, self.feature_selection] @ self.omega[:, self.feature_selection].T + self.gamma, axis=1)

        for h in range(ny):
            ic = self.coeff_index[h]
            if not self.categorical[h]:
                # numerical target
                yhat[:, h] = np.sum(self.coef_[delta, :, ic] * X, axis=1).reshape(N) + self.intercept_[
                    delta, ic].reshape(N)
            else:
                # categorical target
                nic = len(ic)
                aux = np.zeros((N, nic))
                for i in range(nic):
                    aux[:, i] = np.sum(self.coef_[delta, :, ic[i]] * X, axis=1).reshape(N) + self.intercept_[
                        delta, ic[i]].reshape(N)
                imax = np.argmax(aux, axis=1)  # target value is given by argmax of linear predictors
                yhat[:, h] = np.array(self.cat_values[h])[imax]

        if issingle:
            yhat = yhat.reshape(ny)
            delta = delta.reshape(1)

        return yhat, delta

    ############################################
    def score(self, X, Y):
        '''
        PARC - Score function.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        X : float ndarray of shape (number of data, number of features).
            Feature vectors. We assume that categorical features have already been one-hot encoded as 0/1 features.

        Y : float ndarray of shape (number of data, number of targets).
            Target vectors.

        Returns:
        --------
        score : float.
            Compute score of self.predict(X) with respect to Y, one score per target.
            The score is either the R2 score for numeric targets, or accuracy for categorical targets.
        '''

        N = X.shape[0]
        ny = self.ny

        Yt = Y.reshape(N, ny)
        Yhat, _ = self.predict(X)
        Yhat.reshape(N, ny)

        score = np.zeros(ny)
        for j in range(ny):
            if not self.categorical[j]:
                # numeric target
                ymean = np.sum(Yt[:, j]) / N
                score[j] = 1 - np.sum((Yt[:, j] - Yhat[:, j]) ** 2) / np.sum((Yt[:, j] - ymean) ** 2)
            else:
                # categorical target
                score[j] = np.sum(Yt[:, j] == Yhat[:, j]) / N

        return score

    ############################################
    def optimize(self, xmin, xmax, yref, verbose=1, solver="CBC"):
        '''
        PARC - Optimize function.
        (C) 2021-2023 A. Bemporad.

        Given a model with only numeric targets and a target set-point vector yref,
        find the feature vector x such that yhat(x) is closest to yref.

        Use python mip package to model the problem.

        Parameters:
        ----------
        xmin : float ndarray of shape (number of features).
            Lower bound on feature vectors to optimize.

        xmax : float ndarray of shape (number of targets).
            Upper bound on feature vectors to optimize.

        yref : float ndarray of shape (number of targets).
            Reference target vector to achieve.

        Returns:
        --------
        xopt : float ndarray of shape (number of features).
               Optimal feature vector.

        yopt : float ndarray of shape (number of targets).
               Optimal target vector.

        region : int.
               Polyhedral region the optimal feature vector belongs to.

        fopt : float.
               Optimal cost $\| y(xopt)-yref \|_\infty$

        TODO: implement mixed-integer encoding for categorical targets.
        TODO: speedup code to take into account only features such that feature_selection=True
        '''

        if any(self.categorical):
            raise Exception("Only continuous targets supported at the moment.")

        if solver == "CBC":
            solver_name = CBC
        elif solver == "GRB":
            solver_name = GRB
        else:
            raise Exception('Unknown solver. Use either CBC or GRB')

        a = self.coef_
        b = self.intercept_
        omega = self.omega
        gamma = self.gamma

        K, nx, ny = a.shape  # number of clusters, number of features

        m = Model(sense=MINIMIZE, solver_name=solver_name)  # either CBC or GRB for Gurobi
        m.verbose = verbose

        ###################
        # Encode PWL partition

        x = [m.add_var(name="x({})".format(i), var_type=CONTINUOUS,
                       lb=xmin[i], ub=xmax[i]) for i in range(nx)]
        delta = [m.add_var(name="delta({})".format(j), var_type=BINARY)
                 for j in range(K)]
        p = [[m.add_var(name="p({},{})".format(j, i), var_type=CONTINUOUS)
              for i in range(ny)] for j in range(K)]
        y = [m.add_var(name="y({})".format(i), var_type=CONTINUOUS)
             for i in range(ny)]

        # Compute big-M for partition
        M = np.zeros((K, K))
        for j in range(K):
            for i in range(K):
                if not i == j:
                    M[j, i] = sum((max(omega[i, h] - omega[j, h], 0) * xmax[h]
                                   - max(-omega[i, h] + omega[j, h], 0) * xmin[h])
                                  for h in range(nx)) - gamma[j] + gamma[i]

        # Exclusive-or constraint
        #m.add_sos([(delta[i], i) for i in range(K)], 1)
        m.add_constr(xsum(delta[i] for i in range(K)) == 1.0)

        # big-M constraint for PWL partition
        for j in range(K):
            for i in range(K):
                if not i == j:
                    m.add_constr(xsum((omega[i, h] - omega[j, h]) * x[h] for h in range(nx))
                                 <= gamma[j] - gamma[i] + M[j, i] * (1 - delta[j]),
                                 "pwl-j=%d,i=%d" % (j, i))

        # Compute big-M for yc
        Mcp = np.zeros((K, ny))
        Mcm = np.zeros((K, ny))

        for j in range(K):
            for i in range(ny):
                Mcp[j, i] = np.sum(a[j][:, i] * (a[j][:, i] >= 0) * xmax
                                   - (-a[j][:, i] * (a[j][:, i] <= 0) * xmin)) + b[j, i]
                Mcm[j, i] = np.sum(a[j][:, i] * (a[j][:, i] >= 0) * xmin
                                   - (-a[j][:, i] * (a[j][:, i] <= 0) * xmax)) + b[j, i]

        # big-M constraint for yc
        for j in range(K):
            for i in range(ny):
                ax = xsum(a[j][h, i] * x[h] for h in range(nx)) + b[j, i]
                m.add_constr(p[j][i] <= ax - Mcm[j, i] * (1 - delta[j]), "max[0]=%d,i=%d" % (j, i))
                m.add_constr(p[j][i] >= ax - Mcp[j, i] * (1 - delta[j]), "max[1]=%d,i=%d" % (j, i))
                m.add_constr(p[j][i] <= Mcp[j, i] * delta[j], "max[2]=%d,i=%d" % (j, i))
                m.add_constr(p[j][i] >= Mcm[j, i] * delta[j], "max[3]=%d,i=%d" % (j, i))

        # Set yc
        for i in range(ny):
            m.add_constr(y[i] == xsum(p[j][i] for j in range(K)), "y=sum(p)")

        # Minimize |yc-yref|_infty
        eps = m.add_var(name="slack", var_type=CONTINUOUS, lb=0)
        for i in range(ny):
            m.add_constr(eps >= y[i] - yref[i])
            m.add_constr(eps >= -y[i] + yref[i])

        m.objective = eps

        m.optimize()

        fopt = eps.x
        xopt = np.zeros(nx)
        yopt = np.zeros(ny)
        deltaopt = np.zeros(K)
        for i in range(nx):
            xopt[i] = x[i].x
        for i in range(ny):
            yopt[i] = y[i].x
        for i in range(K):
            deltaopt[i] = delta[i].x
        popt = np.zeros((K, ny))
        for j in range(K):
            for i in range(ny):
                popt[j, i] = p[j][i].x

        region = np.where(deltaopt)[0]

        return xopt, yopt, region, fopt

    def plot_partition(self, xmin, xmax, ind=None, values=None, fontsize=12, **kwargs):
        '''
        PARC - Plot piecewise affine partition.
        (C) 2021-2023 A. Bemporad.

        Plot 2D partition if there are only 2 features, otherwise a section for fixed features
        x[ind]=values.

        Parameters:
        ----------
        xmin : float ndarray of shape (2).
            Lower bound on free features plotted.

        xmax : float ndarray of shape (2).
            Upper bound on free features plotted.

        ind: int ndarray of shape (number of features - 2).
            Indices of features that are fixed at given values.
            Default : ind = [2,3,...,nx], i.e., plot PWA partition in (x[0],x[1]) plane.

        values: float ndarray of shape (number of features - 2).
            Values of fixed features. Default: all zeros.

        fontsize: int
            Font used to plot region numbers. Set fontsize=0 for not displaying region numbers.

        **kwargs: arguments passed to matplotlib.collections.PatchCollection

        '''

        nx = self.nx
        if ind is None:
            ind = np.arange(2, nx, dtype=int)
        if values is None:
            values = np.zeros(nx - 2)

        omega = self.omega
        gamma = self.gamma + (omega[:, ind] @ values).ravel()
        omega = np.delete(omega, ind, axis=1)
        xbar = np.delete(self.xbar, ind, axis=1)
        K = self.K

        # Plot PWL partition
        A = np.vstack((np.eye(2), -np.eye(2), np.zeros((K - 1, 2))))
        B = np.vstack((np.array([xmax[0], xmax[1], -xmin[0], -xmin[1]]).reshape(4, 1),
                       np.zeros((K - 1, 1))))
        patches = list()

        for j in range(0, K):
            i = 4
            for h in range(0, K):
                if h != j:
                    A[i, :] = omega[h, :] - omega[j, :]
                    B[i] = -gamma[h] + gamma[j]
                    i += 1

            V = np.array(pypoman.compute_polytope_vertices(A,B))

            # Sort vertices
            cV=np.sum(V,axis=0)/V.shape[0]
            dV=V-cV
            aV=np.arctan2(dV[:,0],dV[:,1])
            iV=np.argsort(aV)
            V=V[iV]

            patches.append(Polygon(V,**kwargs))

            if fontsize > 0:
                plt.text(xbar[j, 0], xbar[j, 1], "%d" % (j + 1), color=(1, 1, 1), fontsize=fontsize)
                plt.text(xbar[j, 0] - (xmax[0] - xmin[0]) * .005, xbar[j, 1] + (xmax[1] - xmin[1]) * .005,
                         "%d" % (j + 1),
                         color=(.1, .1, .1), fontsize=fontsize)

        collection = PatchCollection(patches, **kwargs)
        plt.gca().add_collection(collection)

        return

    def cross_val(self, X_train, Y_train, categorical, weights=None, feature_selection=None,
                  Kfolds=10, Ks=[3, 5], fraction=1, verbose=None):
        '''
        PARC - K-fold cross validation.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        X_train : int ndarray of shape (number of data, number of features).
            We assume that categorical features have already been one-hot encoded as 0/1 features.

        Y_train : int ndarray of shape (number of data, number of targets).
            We assume for each categorical target its categories are encoded as distinct numbers.
            For example, as {0,1} in case of binary targets and {1,2,3,4} for in case of
            4 possible categories.

        categorical : bool ndarray of shape (number of targets).
            Specifies whether a target is numeric (False) or categorical (True).
            For example, if y[0],y[1] are numeric (real-valued), and y[2],y[3] are
            categorical set categorical=[False,False,True,True].

        weights : float ndarray of shape(number of targets)
            weights on output components. Default = 1.

        Kfolds : int.
            Number of folds (default: 10)

        Ks : list of int values.
            List of values (such as a range) of K to cross validate (default: [3,5]).

        fraction : float.
            Fraction of samples actually used for cross validation (default: 1 = all training samples)

        verbose : int
            verbosity level (0 = none, 1 = minimal, 2 = detailed)

        Returns:
        ----------

        best_K : int.
            Best value of K found during cross-validation

        results["mean_score"]: float ndarray of shape(len(Ks)).
            Average score achieved for each tested value of K.

        results["K"]: float ndarray of shape(len(Ks)).
            Tested values of K.

        results["scores"]: float ndarray of shape(len(Ks), Kfolds).
            Individual scores achieved in each of the Kfolds tests, for each tested value of K.
        '''

        def progress_bar(t, T, size=20):
            frac = t / T
            curr = round(size * frac)
            print('\r', '#' * curr + '-' * (size - curr), '[{:>7.2%}]'.format(frac), end='')

        def score_function(K):
            niter = score_function.counter

            K1 = round(K[0])
            ii = (results["K"] == K1)
            if np.any(ii):
                # value of K already tried out
                mean_score = results["mean_score"][ii]
                if verbose > 0:
                    progress_bar(Kfolds, Kfolds)
            else:
                # Run K-fold cross validation

                results["K"][niter] = K1

                selfK = self.K  # save internal K
                self.K = round(K1)
                scores = np.zeros(Kfolds)
                h = 0
                for train_index, test_index in kf.split(X_train):
                    self.fit(X_train[train_index, :], Y_train[train_index, :], categorical,
                             weights=weights, feature_selection=feature_selection)
                    sc = self.score(X_train[test_index], Y_train[test_index, :])
                    scores[h] = np.sum(sc * weights) / self.ny  # weighted average score
                    if verbose > 0:
                        progress_bar(h + 1, Kfolds)
                    h += 1
                mean_score = np.sum(scores) / Kfolds  # average scores on folds

                results["mean_score"][niter] = mean_score
                results["scores"][niter, :] = scores

                score_function.counter += 1
                self.K = selfK  # restore original K

            if verbose > 0:
                print(" K =%3d, mean score = %5.4f" % (K1, mean_score))

            return mean_score

        if verbose is None:
            verbose = 1

        # Initialization
        score_function.counter = 0

        N, nx = X_train.shape
        throwerror = False
        if len(Y_train.shape) == 1:
            Ny = len(Y_train)
            if Ny == N:
                Y_train = Y_train.reshape(N, 1)
            else:
                throwerror = True
        else:
            Ny = Y_train.shape[0]
            if not Ny == N:
                throwerror = True
        ny = Y_train.shape[1]
        if throwerror:
            raise Exception('Number of feature vectors and targets is not consistent.')

        if weights is None:
            weights = np.ones(ny)
        if feature_selection is None:
            feature_selection = np.ones(nx, dtype=bool)

        # Create folds
        kf = KFold(n_splits=Kfolds, random_state=None, shuffle=False)

        # save self parameters
        selfK = self.K
        selfverbose = self.verbose
        self.verbose = int(verbose >= 2)

        if fraction < 1:
            ii = np.random.choice(range(N), round(fraction * N))
            X_train = X_train[ii, :]
            Y_train = Y_train[ii, :]

        nK = len(Ks)
        NN = nK
        results = {"mean_score": np.zeros(NN), "K": np.zeros(NN),
                   "scores": np.zeros((NN, Kfolds))}

        for i in range(nK):
            results["mean_score"][i] = score_function(np.array([Ks[i]]))

        best_K = int(results["K"][np.argmax(results["mean_score"])])

        # restore verbose
        self.verbose = selfverbose

        return best_K, results

    def save(self, filename):
        '''
        PARC - Save PARC predictor to file.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        filename : string filename (without extension) the predictor must be saved to.
        '''

        pickle.dump(vars(self), open(filename, "wb"))

        return

    def load(self, filename):
        '''
        PARC - Load PARC predictor from file.
        (C) 2021-2023 A. Bemporad.

        Parameters:
        ----------
        filename : string filename (without extension) the predictor must be loaded from.
        '''

        saved = pickle.load(open(filename, "rb"))
        keys = list(saved.keys())
        for i in range(len(keys)):
            setattr(self, keys[i], saved[keys[i]])

        return