import abc
import numpy as np

# Based on the MATLAB code from 
# A. Leijon, "Pattern recognition fundamental theory and exercise problems," KTH Elec- tral Engineering, 2010
class ProbDistr(object):
    """
    Probability distribution superclass
    """
    __metaclass__ = abc.ABCMeta

    #### Subclasses must implement these methods ####
    @abc.abstractproperty
    def data_size(self):
        """ Dimension of vector in one sample."""
        return

    @abc.abstractmethod
    def rand(self, nX):
        """
        Generate nX random examples from model self
        """
        return

    @abc.abstractmethod
    def logprob(self, x_data, pD_list):
        """
        log probability of observed data sequence x

        Return
        logP: [n_pD, n_samples], logP[i, j] = log prob of x[j] in pD[i].
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

    def prob(self, x, pD_list=None):
        """
        Default method is the prob method is implemented in the subclass.
        Compute probability of each element in observed data sequence.
        Input:
        -------
        x: 2D array [n_samples, n_features]
        pD_list: list of ProbDistr objects, same class as self
        Return:
        -------
        p: [n_pD, n_samples], p[i, j] = prob of x[j] in pD[i].
        logS: scaling factor, such that the true probability density is pX = p .* exp(logS)
        """
        n_samples = x.shape[0]
        if pD_list is None:
            pD_list = [self]
        logP = self.logprob(x, pD_list) # [n_pD, n_samples]
        if len(pD_list) == 1:
            logS = np.max(logP)
            logP = logP - logS
        else:
            logS = np.max(logP, axis=0) # logS is scalar if self is a single object
            logP = logP - np.tile(logS, (len(pD_list), 1))
        
        logP[np.isnan(logP)] = 0.0
        # correct when logS = -Inf
        if np.isscalar(logS):
            if logS == float('-Inf'):
                p = np.zeros(n_samples)
            else:
                p = np.exp(logP)
        else:
            p = np.exp(logP)
            p[:, logS == float('-Inf')] = 0
        return p, logS
