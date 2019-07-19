"""
This module implements the Lowess function for nonparametric regression.

Functions:
lowess        Fit a smooth nonparametric regression curve to a scatterplot.

For more information, see

William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.

William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Nelle Varoquaux <nelle.varoquaux@gmail.com>
#
# License: BSD (3-clause)

from math import ceil
import numpy as np
from scipy import linalg
import logging

class Lowess(object):
    """Lowess estimator"""

    def __init__(self, f=2./3, niter=3):
        self.f = f
        self.niter = 3

    def fit(self, X, y):
        X = X.flatten()
        y = y.flatten()
        n = len(X)
        r = int(ceil(self.f*n))
        logging.debug("n: %d r: %d" % (n,r))
        logging.debug("y shape: " + str(y.shape))
        logging.debug("y: " + str(y))
        h = [np.sort(np.abs(X - X[i]))[r] for i in range(n)]
        w = np.clip(np.abs((X[:, None] - X[None, :]) / h), 0.0, 1.0)
        w = (1 - w**3)**3
        yest = np.zeros(n)
        delta = np.ones(n)
        betas = np.zeros((n, 2))
        for iteration in range(self.niter):
            for i in range(n):
                weights = delta * w[:, i]
                b = np.array([np.sum(weights*y), np.sum(weights*y*X)])
                A = np.array([[np.sum(weights), np.sum(weights*X)],
                              [np.sum(weights*X), np.sum(weights*X*X)]])
                logging.debug(str(A.shape) + " " + str(b.shape))
                logging.debug(A)
                logging.debug(b)
                beta = linalg.solve(A, b)
                logging.debug("beta: " + str(beta))
                yest[i] = beta[0] + beta[1]*X[i]
                betas[i] = beta

            residuals = y - yest
            logging.debug("y: "+str(y))
            logging.debug("yest: "+str(yest))
            #logging.info("residuals: "+str(residuals))
            s = np.median(np.abs(residuals))
            delta = np.clip(residuals / (6.0 * s), -1, 1)
            delta = (1 - delta**2)**2
        #logging.info("delta: " + str(delta))
        self.delta_ = delta
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X, return_der=False):
        # XXX This is highly unefficient. I refit a hundred times the same
        # model. But I think it works...
        X = X.flatten()
        n = len(self.X_)
        r = int(ceil(self.f*n))

        h_new = [np.sort(np.abs(self.X_ - X[i]))[r] for i in range(len(X))]
        w_new = np.clip(
            np.abs((self.X_[:, None] - X[None, :]) / h_new), 0.0, 1.0)
        w_new = (1 - w_new**3)**3
        yest = np.zeros(len(X))
        derivatives = np.zeros(len(X))
        for i in range(len(X)):
            weights = self.delta_ * w_new[:, i]
            b = np.array([np.sum(weights*self.y_),
                          np.sum(weights*self.y_*self.X_)])
            A = np.array(
                [[np.sum(weights), np.sum(weights*self.X_)],
                 [np.sum(weights*self.X_), np.sum(weights*self.X_*self.X_)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*X[i]
            derivatives[i] = beta[1]
        if return_der:
            return yest, derivatives
        else:
            return yest

    def derivatives(self, X):
        X = X.flatten()
        n = len(self.X_)
        r = int(ceil(self.f*n))

        h_new = [np.sort(np.abs(self.X_ - X[i]))[r] for i in range(len(X))]
        w_new = np.clip(
            np.abs((self.X_[:, None] - X[None, :]) / h_new), 0.0, 1.0)
        w_new = (1 - w_new**3)**3
        yest = np.zeros(len(X))
        derivatives = []
        for i in range(len(X)):
            weights = self.delta_ * w_new[:, i]
            b = np.array([np.sum(weights*self.y_),
                          np.sum(weights*self.y_*self.X_)])
            A = np.array(
                [[np.sum(weights), np.sum(weights*self.X_)],
                 [np.sum(weights*self.X_), np.sum(weights*self.X_*self.X_)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*X[i]
            derivatives.append(beta[1])
        return np.array(derivatives)
