from __future__ import division
import abc
import numpy as np
from prob_distr import ProbDistr

class DiscreteDistr(ProbDistr):
    """
    Class of random discrete integer.
    ...fill in the details...
    """

    def __init__(self, p_mass=None):
        """
        Input:
        p_mass: 1D array or scalar, of probability values, not always normalized.

        Properties:
        -----------
        prob_mass: 1D array of probability values, always normalized.
        """
        self.data_size = 1
        self.pseudo_count = 0
        
        if p_mass is not None:
            self.prob_mass = p_mass if not np.isscalar(p_mass) else np.array(p_mass)
            self.prob_mass = self.prob_mass * 1.0 / np.sum(self.prob_mass) # normalize

    def rand(self, n_data):
        """
        Return 1D array of nX random integers (1 to len(prob_mass)) drawn from the given distribution.
        """
        rdata = np.zeros((n_data))
        r0 = np.random.uniform(0, 1, n_data) # uniform distribution between [0, 1]
        cum_prob = np.cumsum(self.prob_mass)

        for i in range(0, n_data):
            rdata[i] = np.sum([r0[i] > thresh for thresh in cum_prob]) + 1
        return rdata

    def prob(self, Z):
        """
        Probability of integer sequence, drawn from given Discrete distribution.
        Input:
        ------
        Z: [n_samples, 1] or [n_samples, 1], data sequence
        Return:
        ------
        p: 1D array [n_samples, ]. Probability for each element in Z.
        logS: scalar factor, always == 0. True probability = p * exp(logS)
        """
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            raise ValueError("Data must be 1 dimension.")
        Z = np.ravel(Z)
        n_samples = Z.shape[0]
        Z = np.round(Z)
        p = np.zeros((n_samples)) # if Z is out of range, probability is zero
        
        positive_prob_mask = [(Z >= 1) & (Z <= len(self.prob_mass))]
        p[positive_prob_mask] = self.prob_mass[(Z[positive_prob_mask] - 1).tolist()]
        logS = 0.0
        return p, logS

    def logprob(self, x):
        """ log probability mass """
        return np.log(self.prob(x))

    def init_by_data(self, x):
        """
        Initialize discrete distribution model crudely to conform with given data.
        Input:
        -------
        self: a single DiscreteDistr object (may change later).
        x: [n_samples, 1] or [n_samples, 1], data sequence
        """
        if len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError("Data must be 1 dimension.")
        n_samples = x.shape[0]
        x = np.ravel(x)
        x = np.round(x)
        max_obs = np.max(x)
        # observation frequencies
        freq_obs = np.zeros((max_obs))
        for m in range(1, int(max_obs) + 1):
            freq_obs[m - 1] = 1 + np.sum(x == m) # smooth by adding 1, no zero frequencies.
        self.prob_mass = freq_obs / np.sum(freq_obs)

    def adapt_start(self):
        """
        Start DiscreteDistr object adaptation to observed data, by initializing accumulator data structure.
        Input:
        self: a single DiscreteDistr object (may change to list of objects later).
        Return:
        -------
        a_state: the accumulator data structure object
        """
        a_state = DiscreteDAState()
        a_state.sum_weight = np.array([0.0])
        return a_state

    def adapt_accum(self, a_state, obs_data, obs_weight=None):
        """
        Adapt to a single DiscreteDistr object (may change later) by accumulating sufficient statistics from data.
        Input:
        ------
        self: a single DiscreteDistr object (may change later)
        a_state: accumulator data structure object from previous calls
        obs_data: [n_samples, ] or [n_samples, 1]. Observation data sequence.
        obs_weight: same size as obs_data. Default is a vector with all 1's.
        Return:
        ------
        a_state
        """
        if len(obs_data.shape) > 1 and obs_data.shape[1] > 1:
            raise ValueError("Data must be one dimension")
        obs_data = np.ravel(obs_data)
        obs_data = np.round(obs_data)
        max_obs = np.max(obs_data)
        n_samples = obs_data.shape[0]

        if obs_weight is None:
            obs_weight = np.ones((n_samples))

        max_label = max(max_obs, len(self.prob_mass))
        prev_max_label = a_state.sum_weight.shape[0]
        if prev_max_label < max_label:
            # extend size
            a_state.sum_weight = np.concatenate( (a_state.sum_weight, np.zeros((max_label - prev_max_label))) )
        for m in range(1, max_label + 1):
            a_state.sum_weight[m - 1] += np.sum(obs_weight[obs_data == m])
        return a_state

    def adapt_set(self, a_state):
        """
        Finally adapt a DiscreteDistr object using accumulated data.
        Input: self is a single object (may change later)
        """
        a_state.sum_weight += self.pseudo_count * 1.0 / len(a_state.sum_weight)
        self.prob_mass = a_state.sum_weight / np.sum(a_state.sum_weight)


class DiscreteDAState(object):
    def __init__(self):
        self.sum_weight = None
