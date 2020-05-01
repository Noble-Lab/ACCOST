"""
NBfit.py

Module to fit negative binomial means and dispersions to contact count data.

Note that much of this was copied directly or re-written from code provided by Nelle Varoquaux.

"""

import logging
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted



class PolynomialEstimator(LinearRegression):
    """
    Estimator to perform polynomial regression.
    """
    def __init__(self,degree=1,fit_intercept=False):
        assert isinstance(degree,int), "degree should be integer"
        assert isinstance(fit_intercept,bool), "fit_intercept should be boolean"
        self.degree = degree
        self.fit_intercept = fit_intercept
        poly = PolynomialFeatures(degree=self.degree, include_bias=self.fit_intercept)
        self.poly = poly
        # set up defaults for LinearRegression object so it doesn't complain
        self.n_jobs = 1
        self.normalize = False
        self.copy_X = True
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        LinearRegression.fit(self,X_poly,y)
        return self
    
    def predict(self,X):
        X_poly = self.poly.fit_transform(X)
        return LinearRegression.predict(self,X_poly)
            

class LogPolyEstimator(LinearRegression):
    """
    Estimator to perform polynomial regression in log space.
    """
    def __init__(self,degree=2,fit_intercept=True):
        assert isinstance(degree,int), "degree should be integer"
        assert isinstance(fit_intercept,bool), "fit_intercept should be boolean"
        self.degree = degree
        self.fit_intercept = fit_intercept #this is always true for log space regression
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.poly = poly
        # set up defaults for LinearRegression object so it doesn't complain
        self.n_jobs = 1
        self.normalize = False
        self.copy_X = True

    def fit(self, X, y):
        n = np.shape(X)[0]
        X = np.reshape(X,(n,1))
        y = np.reshape(y,(n,1))

        X_poly = self.poly.fit_transform(np.log(X))
        LinearRegression.fit(self,X_poly,np.log(y))
        
        return self

    def predict(self,X):
        X_poly = self.poly.fit_transform(np.log(X))
        return np.exp(LinearRegression.predict(self,X_poly))

