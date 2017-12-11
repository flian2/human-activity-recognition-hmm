from __future__ import division
import abc
import warnings
import numpy as np
from numpy import linalg as LA
from prob_distr import ProbDistr

class GaussDistr(ProbDistr):
    """ Class of Gaussian distribution
    Properties:
    -----------
    Mean: [n_features, ]. array of mean for each feature.
    std: [n_features, 1]. array of standard deviation for each feature.
    variance: [n_features, 1]. array of variance for each feature.
    covariance: [n_features, n_features], covariance matrix
    allow_corr: True, if full covariance matrix allowed. False, if covariance matrix forced to be diagonal.
    cov_eigen: matrix with covariance eigenvectors. cov_eigen * diag(std^2) * cov_eigen^T == covariance
               thus for column vector x: y = cov_eigen^T * x, y is uncorrelated with same variance.
    """

    @property
    def data_size(self):
        """Length of the Gaussian R.V. """
        return len(self.mean)

    @property
    def variance(self):
        return self.std ** 2

    @variance.setter
    def variance(self, var):
        self.std = np.sqrt(var)


    @property
    def covariance(self):
        return np.dot(np.dot(self.cov_eigen, np.diag(self.std ** 2)), np.transpose(self.cov_eigen))

    @covariance.setter
    def covariance(self, c):
        # c is square covariance matrix
        if c.shape[0] != c.shape[1]:
            raise ValueError("Covariance matrix must be square")
        # c must be symmetric
        if np.max(np.abs(c - c.T)) < 1e-4 * np.max(np.abs(c + c.T)):
            eigv, self.cov_eigen = LA.eig(c)
            # force to real components
            self.cov_eigen = np.real(self.cov_eigen)
            self.std = np.sqrt(np.abs(eigv))
        else:
            raise ValueError("Covariance matrix must be symmetric")

    @property
    def allow_corr(self):
        return not np.isscalar(self.cov_eigen) # if cov_eigen is scalar, allow_cor is false

    @allow_corr.setter
    def allow_corr(self, ac):
        if ac is True:
            if not self.allow_corr:
                self.cov_eigen = np.eye((len(self.mean)))
        else:
            # force covariance to be diagonal
            if self.allow_corr:
                self.cov_eigen = 1.0

    def __init__(self, mean=None, std=None, covariance=None):
        self.mean = mean
        self.std = std
        self.cov_eigen = 1.0
        if covariance is not None:
            self.covariance = covariance
    
    def rand(self, n_samples):
        """
        Generate random vectors from the given Gaussian distribution.
        Input: 
        ------
        n_samples: length of random vectors generated.
        Return:
        x: [n_samples, n_features]
        """
        n_features = self.data_size
        x = np.random.normal(0, 1, (n_samples, n_features)) # normal distributin
        # y = x * (diag(std) * cov_eigen^T)
        x = x.dot(np.dot(np.diag(self.std), np.transpose(self.cov_eigen))) # scaled to correct correlation
        x = x + np.tile(self.mean, (n_samples, 1)) # shifted to correct mean
        return x

    def logprob(self, x):
        """
        Log probability for given vectors.
        Input:
        ------
        x: [n_samples, n_features]
        Return:
        logP: [n_samples, ]. The log probability of x. 
        """
        n_samples, n_features = x.shape
        if n_features != self.data_size:
            raise ValueError("The Gaussian Distribution has size %d, expecting same size in x" % self.data_size)
        logP = np.zeros((n_samples))
        z = np.dot((x - np.tile(self.mean, (n_samples, 1))), self.cov_eigen) # transform to uncorrelated samples.
        z = z / np.tile(self.std, (n_samples, 1))
        # pdf: (2pi*det(C))^-1/2 * exp(-1/2*[(x-mu)^T * C^{-1} * (x-mu)])
        logP[:] = - np.sum(z ** 2, axis=1) / 2.0 # normalized Gaussian exponent
        logP[:] = logP[:] - np.sum(np.log(self.std)) - self.data_size * np.log(2 * np.pi) / 2.0
        return logP

    def init_by_data(self, x):
        """
        Crude initializaiton of a single GaussDistr object to conform with given data.
        Input:
        ------
        x: [n_samples, n_features] or [n_samples, ]. Observed data sequence.

        Return:
        ------
        is_ok: True if GaussDistr is properly initialized, False if not enough data for good initialization.

        Method:
        ------
        Set mean and variance, based on all observations. Preserve previous allow_corr, but only set the 
        diagonal values of covariance matrix.

        """
        n_samples, n_features = x.shape
        if n_samples <= 1:
            # not enough data to estimate variance
            warnings.warn("Only one data point, default variance = 1.")
            var_x = 1.0
            is_ok = False
        else:
            var_x = np.var(x, axis=0) # biased estimation of variance.
            is_ok = True
        # Initialize to observation sub-set. 
        # Use VectorQuantizer initialization:
        # set mean and variance at VQ cluster centers, and set variance to variance within each VQ center.
        self.mean = np.mean(x, axis=0)
        if self.allow_corr is True:
            self.covariance = np.diag(var_x)
        else:
            self.variance = var_x
        return is_ok
    
    def adapt_start(self):
        return

    def adapt_accum(self):
        return

    def adapt_set(self):
        return





