import abc
import numpy as np

class ProbDistr(object):
    """
    Probability distribution superclass
    """
    __metaclass__ = abc.ABCMeta

    #### Subclasses must implement these methods ####

    @abc.abstractmethod
    def rand(self, nX):
        """
        Generate nX random examples from model self
        """
        return

    @abc.abstractmethod
    def logprob(self, x_data):
        """
        log probability of observed data sequence x

        Return log prob of each element in the sequence x.
        """
        return

    @abc.abstractmethod
    def init_by_data(self, x):
        """
        Initialize probability distribution model crudely to conform with given data.
        """
        return

    @abc.abstractmethod
    def adapt_start(self):
        return

    @abc.abstractmethod
    def adapt_accum(self, a_state, obs_data):
        return

    @abc.abstractmethod
    def adapt_set(self, a_state):
        return

    ###################################################

    def prob(self, x):
        """
        Default method is no prob method is implemented in the subclass.
        Compute probability of each element in observed data sequence.
        Input:
        -------
        x: 2D array [n_samples, n_features]
        Return:
        -------
        p: 1D array [n_samples, ]. probability of each sample x in the sequence.
        logS: Scalar scaling factor, such that the true probability density is pX = p * exp(logS)
        """
        n_samples = x.shape[0]
        logP = self.logprob(x)
        logS = np.max(logP) # logS is scalar if self is a single object
        logP = logP - logS
        logP[np.isnan(logP)] = 0.0
        # correct when logS = -Inf
        if logS == float('-Inf'):
            p = np.zeros(n_samples)
        else:
            p = np.exp(logP)
        return p, logS
