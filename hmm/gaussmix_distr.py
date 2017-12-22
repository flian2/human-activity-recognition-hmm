from __future__ import division
import abc
import warnings
import numpy as np
from numpy import linalg as LA
from prob_distr import ProbDistr
from gauss_distr import GaussDistr
from discrete_distr import DiscreteDistr
from vector_quantizer import *

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
        """
        Usage:
        gmm = GaussMixDistr()
        gmm = GaussMixDistr(4);
        gmm = GaussMixDistr(list_of_gd, m_w)
        """
        if gauss is None:
            self._gaussians = [GaussDistr()]
        elif np.isscalar(gauss) and gauss > 0 and np.mod(gauss, 1) == 0:
            # if gauss is positive integer: gauss is n_mixtures
            self._gaussians = [GaussDistr() for i in range(0, gauss)]
        elif isinstance(gauss, list):
            self._gaussians = gauss
        else:
            raise ValueError("First input must be positive integer or list of GaussDistr objects")

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
        rand_int: vector of indices of randomly chosen GuassDistr objects. Can take 1 ~ len(gaussians)
        """
        # Generate random integer sequence according to discrete distribution with pmass mix_weight.
        rand_int = DiscreteDistr(self.mix_weight).rand(n_samples)
        n_features = self.data_size
        x = np.zeros((n_samples, n_features))
        for i in range(1, len(self.gaussians) + 1):
            x[rand_int == i, :] = self.gaussians[i - 1].rand(np.sum(rand_int == i))
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
        to do: do I need to implement init_by_data for multiple gmms?
        """
        n_samples, n_features = x.shape
        if n_samples < 2:
            raise ValueError("Too few data")
        # Init multiple gaussians by clustering
        var_x = np.var(x, axis=0) # [n_features, ]
        n_mixtures = len(self.gaussians)
        vq, _ = make_VQ(x, n_mixtures)
        x_centers = vq.code_book
        x_codes = vq.encode(x)
        for i in range(0, n_mixtures):
            n_data = np.sum(x_codes == i) # number of samples for this centroid
            if n_data <= 1:
                warnings.warn("Too few data for %d-th GaussDistr" % i)
            else:
                var_x = np.var(x[x_codes == i, :], axis=0)
            self.gaussians[i].mean = x_centers[i, :]
            # Only use the diagonal variance
            if self.gaussians[i].allow_corr:
                self.gaussians[i].covariance = np.diag(var_x)
            else:
                self.gaussians[i].variance = var_x
        self.mix_weight = np.ones((n_mixtures)) / n_mixtures

    def adapt_start(self, pD_list):
        """
        Start GaussMixDistr object adaptation to observed data, by initializing accumulator data structure.
        Input:
        pD_list: a list of GuassMixDistr objects.
        Return:
        a_state_list: the list of accumulator data structure object, same size of pD
        """
        a_state_list = [GmmAState() for i in range(0, len(pD_list))]
        n_features = self.data_size
        for i in range(0, len(pD_list)):
            a_state_list[i].gaussians = pD_list[i].gaussians[0].adapt_start(pD_list[i].gaussians)
            a_state_list[i].mix_weight = np.zeros(pD_list[i].mix_weight.shape)
        return a_state_list

    def adapt_accum(self, pD_list, a_state_list, obs_data, obs_weight=None):
        """
        Adapt to a list of GaussMixDistr objects by accumulating sufficient statistics from data.
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

        for i in range(0, len(pD_list)):
            n_mixtures = len(pD_list[i].gaussians)
            if n_mixtures == 1:
                a_state_list[i].gaussians = pD_list[i].gaussians[0].adapt_accum(
                    pD_list[i].gaussians, a_state_list[i].gaussians, obs_data, obs_weight[i, None, :])
                a_state_list[i].mix_weight += np.sum(obs_weight[i, :])
            else:
                # prob for each gaussian distribution.
                # sub_prob: [n_mixtures, n_samples]. 
                # sub_prob[j, t] = P(X(t) | S(t) = i, subS(t) = j)
                sub_prob, _ = pD_list[i].gaussians[0].prob(obs_data, pD_list[i].gaussians) #[n_mixtures, n_samples]
                sub_prob = np.diag(pD_list[i].mix_weight).dot(sub_prob)
                # normalize sub_prob to valid conditional prob
                # sub_prob[j, t] = P(S(t) = i | X(t), subS(t) = j)
                denom = np.maximum(np.spacing(1), np.sum(sub_prob, axis=0)) # [n_samples, ]. avoid zero denominator
                sub_prob = sub_prob / np.tile(denom, (n_mixtures, 1))
                # Scale by externally given weights
                # sub_prob[j, t] = P(S(t) = i, subS(t) = j | X(t) for t = 0,...T)
                sub_prob = np.multiply(sub_prob, np.tile(obs_weight[i, :], (n_mixtures, 1)))
                a_state_list[i].gaussians = pD_list[i].gaussians[0].adapt_accum(
                    pD_list[i].gaussians, a_state_list[i].gaussians, obs_data, sub_prob)
                a_state_list[i].mix_weight += np.sum(sub_prob, axis=1)

        return a_state_list

    def adapt_set(self, pD_list, a_state_list):
        """
        Finally adapt a GaussMixDistr object using accumulated data.
        Input:
        -------
        pD_list: list of GaussMixDistr objects used in the adaptation.
        a_state_list: list of accumulator data structure from previous calls of adapt_accum
        Return:
        -------
        pD_list: list of GaussDistr objects
        Method:
        -------
        Adjust the mix_weight vector by normalizing the accumulated sum of mix_weight.
        """
        for i in range(0, len(pD_list)):
            pD_list[i].gaussians = pD_list[i].gaussians[0].adapt_set(
                pD_list[i].gaussians, a_state_list[i].gaussians)
            pD_list[i].mix_weight = a_state_list[i].mix_weight / np.sum(a_state_list[i].mix_weight)

        return pD_list

    def train(self, obs_data, n_iter=10, min_step=float('Inf')):
        """
        Adapt a single GaussMixDistr object to the given observation data.
        Input:
        ------
        obs_data: Training data. [n_samples, n_features]. 
        n_iter: min number of iterations.
        min_step: min logprob improvement per training observation vector, 
                  = desired improvement in relative entropy per obs vector in each iteration.
        Return:
        ------
        logprobs: values of logprob of training obs set from each iteration. [n_final_iter, ]. 
                  n_final_iter >= n_iter
        """
        logP_old = float("-Inf")
        logP = np.mean(self.logprob(obs_data))
        logprobs = []
        pD_list = [self]
        for n in range(0, n_iter):
            logP_old = logP
            aS = self.adapt_start(pD_list)
            aS = self.adapt_accum(pD_list, aS, obs_data)
            pD_list = self.adapt_set(pD_list, aS)
            logP = np.mean(self.logprob(obs_data), axis=1)
            logprobs.append(logP)

        while logP - logP_old > min_step:
            # continue training if sufficiently good improvement
            logP_old = logP
            aS = self.adapt_start(pD_list)
            aS = self.adapt_accum(pD_list, aS, obs_data)
            pD_list = self.adapt_set(pD_list, aS)
            logP = np.mean(self.logprob(obs_data), axis=1)
            logprobs.append(logP)

        return np.array(logprobs)

    def plot_mixture_centroids(self, pD_list, colors):
        """
        Visualize GaussMixDistr objects on a 2D plot
        Input: 
        ---------
        gaussians: list of GaussMixDistr objects.
        colors: list of colors. Different color for different GaussMixDistr objects, 
        same color among gaussians within one mixture
        """
        n_obj = len(pD_list)
        for i in range(0, n_obj):
            gaussians = pD_list[i].gaussians
            gaussians[0].plot_mixture_centroids(gaussians, [colors[i]])


class GmmAState(object):
    def __init__(self):
        self.gaussians = None
        self.mix_weight = None

def make_gmm_single(mean_s, std_s, m_w=None):
    gaussians = []
    n_mixtures = mean_s.shape[0]
    if m_w is None:
        m_w = np.ones((n_mixtures))
    for i in range(0, mean_s.shape[0]):
        gaussians.append( GaussDistr(mean=mean_s[i, :], std=std_s[i, :]) )
    gmm = GaussMixDistr(gaussians, m_w)
    return gmm
