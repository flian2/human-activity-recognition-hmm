from __future__ import division
import abc
import warnings
import numpy as np
from numpy import linalg as LA
from prob_distr import ProbDistr
from gauss_distr import GaussDistr
from discrete_distr import DiscreteDistr

class GaussMixDistr(ProbDistr):
    """ 
    Gaussian Mixture Model
    Properties:
    -----------
    gaussians: list of GaussDistr objects
    mix_weight: array with weight factors for pdf mix
    mean: [n_features, ]. mean vector of the mixtures
    covariance: [n_features, n_features]. covariance matrix for the deviation of mixtures from the total mean
    """

    @property
    def data_size(self):
        # size of the vector sample of GaussianDistr in gaussians
        return self.gaussians[0].data_size

    @property
    def mean(self):
        m_k = np.zeros((self.data_size, len(self.gaussians)))
        for k in range(0, len(self.gaussians)):
            m_k[:, k] = self.gaussians[k].mean
        return m_k.dot(self.mix_weight[:, np.newaxis])

    @property
    def covariance(self):
        w_mean = self.mean
        C = np.zeros((self.data_size, self.data_size))
        for k in range(0, len(self.gaussians)):
            dev = self.gaussians[k] - w_mean # component deviation from grand mean
            C += self.mix_weight[k] * \
                (self.gaussians[k].covariance + dev[:, np.newaxis].dot(dev[np.newaxis, :]))
        return C

    @property
    def gaussians(self):
        return self._gaussians

    @gaussians.setter
    def gaussians(self, gs):
        if isinstance(gs, GaussDistr):
            self._gaussians = [gs]
        elif isinstance(gs, list):
            for i in range(0, len(gs)):
                if not isinstance(gs[i], GaussDistr):
                    raise ValueError("The %d-th element in the list is not GuassDistr" % i)
                if i > 0 and gs[i].data_size != data_sz:
                    raise ValueError("Each Gaussian component must have the same data size")
                data_sz = gs[i].data_size
            self._gaussians = gs
        else:
            raise ValueError("Input must be GaussDistr object or a list of GaussDistr objects")

    @property
    def mix_weight(self):
        return self._mix_weight

    @mix_weight.setter
    def mix_weight(self, m_w):
        # just normalize
        self._mix_weight = m_w / float(np.sum(m_w))

    def __init__(self, gauss=None, mix_w=None):
        if gauss is None:
            self._gaussians = [GaussDistr()]
        else:
            self._gaussians = gauss
        if mix_w is None:
            self._mix_weight = np.ones((len(self._gaussians)))
        elif len(mix_w) == len(self._gaussians):
            self.mix_weight = mix_w
        else:
            raise ValueError("The length of mix_weight must be equal to the length of gaussians")

    def rand(self, n_samples):
        """
        Generate random vectors from the given Gaussian mixture model.
        Input: 
        ------
        n_samples: length of random vectors generated.
        Return:
        x: [n_samples, n_features]
        rand_int: vector of indices of randomly chosen GuassDistr objects. Can take 0 ~ (len(gaussians) - 1)
        """
        # Generate random integer sequence according to discrete distribution with pmass mix_weight.
        rand_int = DiscreteDistr(self.mix_weight).rand(n_samples) - 1
        n_features = self.data_size
        x = np.zeros((n_samples, n_features))
        for i in range(0, len(self.gaussians)):
            x[rand_int == i, :] = self.gaussians[i].rand(np.sum(rand_int == i))
        return x, rand_int

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
            raise ValueError("The GMM has data size %d, expecting same size in x" % self.data_size)
        if pD_list is None:
            pD_list = [self]
        n_pD = len(pD_list)
        logP = np.zeros((n_pD, n_samples))
        for i in range(0, n_pD):
            # logprob for each mixture
            logP_gs = pD_list[i].gaussians[0].logprob(x, pD_list[i].gaussians) # [n_mixtures, n_samples]
            logS = np.max(logP_gs, axis=0) # 1D array [n_samples, ]. If n_mixtures is 1, all elements are the same
            logP_gs = logP_gs - np.tile(logS, (logP_gs.shape[0], 1))
            logP_gs[np.isnan(logP_gs)] = 0.0
            logP[i, :] = logS + np.log( pD_list[i].mix_weight[np.newaxis, :].dot( np.exp(logP_gs) ) )[0, :] # 1D array

        return logP

    def init_by_data(self, x):
        """
        Crude initializaiton of a single GaussMixDistr object to conform with given data.
        Input:
        ------
        x: [n_samples, n_features] or [n_samples, ]. Observed data sequence.

        Return:
        ------
        is_ok: True if GaussDistr is properly initialized, False if not enough data for good initialization.

        Method:
        ------
        Use equal mix_weight.
        For each GuassDistr object initialized to observation subset. 
        Set mean at VQ cluster center.
        Set variance to variance within each VQ.

        """
        n_samples, n_features = x.shape
        if n_samples < 2:
        	raise ValueError("Too few data")
        # init multiple gaussians by clustering
        # to implement ...
        xVQ = Create(VQ)
        return

    def adapt_start(self):
        return

    def adapt_accum(self):
        return

    def adapt_set(self):
        return

    