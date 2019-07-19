"""
NBfit.py

Module to fit negative binomial means and dispersions to contact count data.

Note that much of this was copied directly or re-written from code provided by Nelle Varoquaux.

<<citation>>
"""


import logging
import numpy as np
from scipy import sparse
#import dispersion
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from lowess_ import Lowess
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from sklearn.utils.validation import check_is_fitted


def get_means_variances(counts,binpairs_reversed,lengths_reversed):
    """
    Given a contact count matrix, get the mean and variance of contact counts at each length.

    Args:
       counts: a matrix of contact counts in dictionary format
       binpairs_reversed: list of bin pair indices pointing to chr/mid tuple
       lengths_reversed: dictionary of lengths to bin pair indices
    Returns:
       means: dictionary of length --> mean
       variances: dictionary of length --> variance
    """
    means = {}
    variances = {}
    logging.info(lengths_reversed.keys()[1:10])
    for length in lengths_reversed.keys():
        indices = lengths_reversed[length]
        lengthcounts = []
        #logging.info(length)
        #logging.info(indices)
        for i in indices:
            (chr1,mid1,chr2,mid2) = binpairs_reversed[i]
            count = counts[chr1][mid1][chr2][mid2]
            #TODO: some filter for counts here (ie do we include zeros?)
            if count is not None:
                lengthcounts.append(count)
        #TODO: filter for minimum number of counts required?
        if len(lengthcounts)>2:
            mean = np.mean(np.array(lengthcounts))
            variance = np.var(np.array(lengthcounts))
            means[length] = mean
            variances[length] = variance
    return(means,variances)

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
        #print(fit_intercept)
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
        logging.debug(X)
        logging.debug(y)
        logging.debug(np.log(X))
        X_poly = self.poly.fit_transform(np.log(X))
        logging.debug(X.shape)
        logging.debug(y.shape)
        logging.debug(X_poly.shape)
        LinearRegression.fit(self,X_poly,np.log(y))
        return self

    def predict(self,X):
        #logging.debug(X)
        #logging.debug(np.log(X))
        X_poly = self.poly.fit_transform(np.log(X))
        return np.exp(LinearRegression.predict(self,X_poly))



class LowessEstimator(BaseEstimator):
    """
    A Lowess estimator that pretends to work like a sklearn estimator.
    """
    def __init__(self, f=2./3, niter=3, logspace=True):
        self.f = f
        self.niter = niter
        self.clf = Lowess(f=f, niter=niter)
        self.logspace = logspace
        

    def fit(self, X, y):
        if self.logspace:
            self.clf.fit(np.log(X), np.log(y))
        else:
            self.clf.fit(X, y)
        return self
    
    def predict(self, X):
        if self.logspace:
            y_ = np.exp(self.clf.predict(np.log(X)))
        else:
            y_ = self.clf.predict(X)
        return y_
    
    def score(self, X, y):
        yest = self.predict(X)
        return r2_score(y, yest, sample_weight=None)


class LocfitEstimator(BaseEstimator):
    """
    An R locfit estimator that pretends to work like a sklearn estimator.
    """
    def __init__(self):
        rfile = "/net/noble/vol2/home/katecook/proj/2015HiC-differential/src/call_locfit.R"
        rfh = open(rfile,'r')
        string = ''.join(rfh.readlines())
        self.locfit = SignatureTranslatedAnonymousPackage(string,"locfit")
    
    def fit(self, X, y):
        X_robj = robjects.FloatVector(X)
        y_robj = robjects.FloatVector(y)
        self.fit_ = self.locfit.do_fit(X_robj,y_robj)

    def predict(self, X):
        check_is_fitted(self,"fit_")
        shape = X.shape
        flat = X.flatten().T
        X_robj = robjects.FloatVector(flat)
        y_robj = self.locfit.safepredict(self.fit_, X_robj)
        y_ = np.array(y_robj).T.reshape(shape)
        #np.savetxt("y.txt",y_,delimiter='\t')
        return y_
    
    def score(self, X, y):
        yest = self.safepredict(X)
        return r2_score(y, yest, sample_weight=None)



#old, do not use
def estimate_dispersion(means,vars):
    raise NotImplementedError("This is wrong now!")
    lengths = sorted(means.keys())
    logging.info(['lengths',lengths[1:10]])
    logging.info(['means',[means[l] for l in lengths[1:10] ]])
    logging.info(['vars',[vars[l] for l in lengths[1:10] ]])
    est = dispersion.estimate_alpha_mean_variance(
        [ means[l] for l in lengths ],
        [ vars[l] for l in lengths ],
        estimator="ols")
    try:
        alpha = est.estimator_.coef_[0]
    except AttributeError: #for lowess estimator
        alpha = est.coef_[0]
    logging.info(alpha)
    dispersion_ = 1 / alpha
    return dispersion_
