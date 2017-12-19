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
        if self.mean is not None:
            return len(self.mean)
        else:
            return 0

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
        x = np.random.normal(0, 1, (n_samples, n_features)) # normal distribution
        # y = x * (diag(std) * cov_eigen^T)
        x = x.dot(np.dot(np.diag(self.std), np.transpose(self.cov_eigen))) # scaled to correct correlation
        x = x + np.tile(self.mean, (n_samples, 1)) # shifted to correct mean
        return x

    def logprob(self, x, pD_list=None):
        """
        Log probability for given vectors.
        Input:
        ------
        x: [n_samples, n_features]
        Return:
        logP: [n_pD, n_samples]. The log probability of x. 
        """
        n_samples, n_features = x.shape
        if n_features != self.data_size:
            raise ValueError("The Gaussian Distribution has size %d, expecting same size in x" % self.data_size)
        if pD_list is None:
            pD_list = [self]
        n_pD = len(pD_list)
        logP = np.zeros((n_pD, n_samples))
        for i in range(0, n_pD):
            z = np.dot((x - np.tile(pD_list[i].mean, (n_samples, 1))), pD_list[i].cov_eigen) # transform to uncorrelated samples.
            z = z / np.tile(pD_list[i].std, (n_samples, 1))
            # pdf: (2pi*det(C))^-1/2 * exp(-1/2*[(x-mu)^T * C^{-1} * (x-mu)])
            logP[i, :] = - np.sum(z ** 2, axis=1) / 2.0 # normalized Gaussian exponent
            logP[i, :] = logP[i, :] - np.sum(np.log(pD_list[i].std)) - pD_list[i].data_size * np.log(2 * np.pi) / 2.0
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
    
    def adapt_start(self, pD_list):
        """
        Start DiscreteDistr object adaptation to observed data, by initializing accumulator data structure.
        Input:
        pD_list: a list of DiscreteDistr objects.
        Return:
        a_state_list: the list of accumulator data structure object, same size of pD
        """
        a_state_list = [GaussDAState() for i in range(0, len(pD_list))]
        n_features = self.data_size
        for i in range(0, len(pD_list)):
            a_state_list[i].sum_dev = np.zeros((n_features))
            a_state_list[i].sum_sq_dev = np.zeros((n_features, n_features))
            a_state_list[i].sum_weight = 0.0
        return a_state_list

    def adapt_accum(self, pD_list, a_state_list, obs_data, obs_weight=None):
        """
        Adapt to a list of GaussDistr objects by accumulating sufficient statistics from data.
        Input:
        -------
        pD_list: a list of GaussDistr objects
        obs_data: [n_samples, ] or [n_samples, 1]. Observation data sequence.
        obs_weight: [n_pD, n_samples] Default is a vector with all 1's.
        Return:
        -------
        a_state_list
        """
        n_samples, n_features = obs_data.shape
        n_pD = len(pD_list)
        if obs_weight is None:
            if n_pD == 1:
                obs_weight = np.ones((n_pD, n_samples)) # all data with equal weight
            else:
                obs_weight = self.prob(obs_data, pD_list) # [n_pD, n_samples]
                # Normalize for each sample: obs_weight(i, t) = P[ obj(t) = i | x(t) ]
                obs_weight = obs_weight / np.tile(np.sum(obs_weight, axis=0), (n_pD, 1))
        for i in range(0, n_pD):
            dev = obs_data - np.tile(pD_list[i].mean, (n_samples, 1)) # deviations from old mean, [n_samples, n_features]
            w_dev = np.multiply( dev, np.tile(obs_weight[i, None, :].T, (1, n_features)) ) # weighted deviations, [n_samples, n_features]
            a_state_list[i].sum_dev += np.sum(w_dev, axis=0) # [n_features, ]
            a_state_list[i].sum_sq_dev += np.dot(dev.T, w_dev)
            a_state_list[i].sum_weight += np.sum(obs_weight[i, :])

        return a_state_list

    def adapt_set(self, pD_list, a_state_list):
        """
        Finally adapt a GaussDistr object using accumulated data.
        Input:
        -------
        pD_list: list of GaussDistr objects used in the adaptation.
        a_state_list: list of accumulator data structure from previous calls of adapt_accum
        Return:
        -------
        pD_list: list of GaussDistr objects

        Method:
        -------
        For the sampled vector X(n), the deviation Z(n) = X(n) - old mean.
        Then E(Z(n)) = E(x(n)) - old mean, cov(Z(n)) = cov(X(n))

        We have the accumulated weighted deviations:
        sum_dev = sum[w(n) * Z(n)] over n
        sum_sq_dev = sum[w(n) * Z(n) * Z(n)^T] over n
        sum_weight = sum[w(n)] over n
        where w(n) is the probability of X(n) drawn from given GaussDistr.

        From E[ sum_dev ] = sum[w(n) * E(Z(n))] = sum[w(n)] * E(z(n))
        Thus an unbiased estimate of mean:
        new mean = old mean + sum_dev / sum_weight
        ML covariance estimation:
            The deviations from the estimated mean:
            Y(n) = Z(n) - sum_dev / sum_weight
            S2 = sum[w(n) * Y(n) * Y(n)^T]
               = sum_sq_dev - sum_dev*sum_dev'./sum_weight
            Thus, the estimate of covariance matrix = S2 ./ sum_weight
        """
        for i in range(0, len(pD_list)):
            # make sure there is some accumulated data
            if a_state_list[i].sum_weight > np.max(np.spacing(pD_list[i].mean)):
                # update mean
                pD_list[i].mean += a_state_list[i].sum_dev / a_state_list[i].sum_weight

                # update covariance
                S2 = a_state_list[i].sum_sq_dev \
                    - a_state_list[i].sum_dev[:, np.newaxis].dot(a_state_list[i].sum_dev[np.newaxis, :]) / a_state_list[i].sum_weight
                cov_estim = S2 / a_state_list[i].sum_weight
                if np.any(np.diag(cov_estim) < np.spacing(pD_list[i].mean)):
                    warnings.warn("Not enough data for GaussDistr object %d, forcing std to zero" \
                        % i )
                    cov_estim = np.diag(np.tile(np.inf, (pD_list[i].data_size)))
                if pD_list[i].allow_corr:
                    pD_list[i].covariance = cov_estim
                else:
                    pD_list[i].std = np.sqrt(np.diag(cov_estim))

        return pD_list


class GaussDAState(object):
    """
    sum_dev: weighted sum of observed deviations from old mean.
    sum_sq_dev: matrix with sum of square deviations from old mean.
    sum_weight: sum of weight factors.
    """
    def __init__(self):
        self.sum_dev = None
        self.sum_sq_dev = None
        self.sum_weight = None
