from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
import copy
from markov_chain import MarkovChain
from prob_distr import ProbDistr


class HMM(object):
    """
    Hidden Markov Model class
    state_gen: a state sequence generator of type MarkovChain
    output_distr: an array of output probability distributions, one for each state
    Dependent properties:
    ---------------------
    n_states: number of Markov chain states
    data_size: length of vectors in the output sequence
    """
    def __init__(self, markov_chain=MarkovChain(), output_distr=None):
        # to do: change default output_distr to GuassD or DiscreteD 
        self._state_gen = markov_chain
        self._output_distr = output_distr

    @property
    def n_states(self):
        return self.state_gen.n_states

    @property
    def data_size(self):
        return self.output_distr[0].data_size # determined by output distribution object

    @property
    def state_gen(self):
        return self._state_gen

    @state_gen.setter
    def state_gen(self, mc):
        # Check type, must be markov chain object
        if isinstance(mc, MarkovChain):
            self._state_gen = mc
        else:
            raise ValueError("Must be a MarkovChain object")

    @property
    def output_distr(self):
        return self._output_distr

    @output_distr.setter
    def output_distr(self, pD):
        # to do: check type after I decide which object to use
        if not isinstance(pD, list):
            if isinstance(pD, ProbDistr):
                self._output_distr = pD
            else:
                raise ValueError("Must be a ProbDistr object")
        else:
            d_size = pD[0].data_size
            if all( pD[i].data_size == d_size for i in range(0, len(pD))):
                self._output_distr = pD 
            else:
                raise ValueError("All distribution must have the same data size")

    def init_leftright_outputdistr(self, obs_data, l_data):
        """
        Initionlize hmm.output_distr crudely according to observation data for left-hmm
        obs_data: [n_samples, n_features]. The concatenated training sequences.
        l_data: [n_sequence, ]. Length of subsequences of the training data. sum(l_data) = n_samples
        """

        # Normally the self.output_distr is a dummy ProbDistr object with data members not assigned.
        # Check input size
        if l_data.sum() != obs_data.shape[0]:
            raise ValueError("Training data has %d samples, expecing the same number in l_data" % obs_data.shape[0])
        # Make output_distr a vector of the same size of hmm states
        if not isinstance(self.output_distr, list):
            self.output_distr = [copy.deepcopy(self.output_distr) for i in range(0, self.n_states)]

        n_samples, n_features = obs_data.shape
        n_sequences = len(l_data)
        start_ind = np.append(np.array([0]), np.cumsum(l_data))
        n_states = self.n_states
        # Use average length length of each state to initialize output distribution
        for i in range(0, n_states):
            data_per_state = np.zeros((0, n_features))
            for r in range(0, n_sequences):
                # staring point for the i-th state in the r-th subsequence
                d_start = start_ind[r] + (i * l_data[r]) / n_states
                d_end = start_ind[r] + ((i+1) * l_data[r]) / n_states # exclusive
                data_per_state = np.concatenate((data_per_state, obs_data[d_start:d_end, :]), axis=0)
            # Very crude initialization, should be refined by training
            self.output_distr[i].init_by_data(data_per_state)

    def train(self, obs_data, l_data, n_iter=10, min_step=float('Inf')):
        """
        Train a single hmm to an observed sequence.
        Input:
        ------
        obs_data: [n_samples, n_features]. The concatenated training sequences. One sample of observed data vector is stored row-wise.
        l_data: [n_sequence, ]. l_data[r] is the length of rth training sequence.
        n_iter: min number of iterations
        min_step: min logprob improvement per training observation vector, 
                  = desired improvement in relative entropy per obs vector in each iteration.
        Return:
        --------
        logprobs: values of logprob of training obs set from each iteration.
        
        Methods: apply methods adapt_start, adapt_accum, adapt_set. HMM must be already initialized.
        """
        if n_iter <= 0:
            raise ValueError("Number of iterations cannot be 0")
        if min_step < float('Inf'):
            min_step = min_step * obs_data.shape[0]

        ixT = np.append(np.array([0]), np.cumsum(l_data)) # start index for each subsequence
        logprobs = [0.0 for i in range(0, n_iter)]
        logP_old = float('-Inf')
        logP_delta = float('Inf') # logP improvement in last step
        for n_training in range(0, n_iter):
            aS = self.adapt_start()
            for r in range(0, len(l_data)):
                aS, logP = self.adapt_accum(aS, obs_data[ixT[r]: ixT[r+1], :])
            logprobs[n_training] += logP
            logP_delta = logprobs[n_training] - logP_old
            logP_old = logprobs[n_training]
            self.adapt_set(aS)
        # Continue training if sufficiently good improvement
        while logP_delta > min_step:
            n_training += 1
            logprobs.append(0.0)
            aS = self.adapt_start()
            for r in range(0, len(l_data)):
                aS, logP = self.adapt_accum(aS, obs_data[ixT[r]: ixT[r+1], :])
            logprobs[n_training] += logP
            logP_delta = logprobs[n_training] - logP_old
            logP_old = logprobs[n_training]
            self.adapt_set(aS)

    def adapt_start(self):
        """
        Initialize adaptation data structure for a single HMM object, to be saved between subsequent calls to adapt_accum.
        Return:
        ------
        a_state: object representing zero weight of previous observed data
        a_state.MC: for hmm.state_gen sub-object
        a_state.Out: for hmm.output_distr sub-object
        a_state.LogProb: for the accumulated log(prob(observations))
        """
        a_state_mc = self.state_gen.adapt_start()
        a_state_out = self.output_distr[0].adapt_start(self.output_distr)
        a_state = AState(a_state_mc, a_state_out, 0)
        return a_state

    def adapt_accum(self, a_state, obs_data):
        """
        Method to adapt to single HMM object to observed data, by accumulating sufficient statistics from the data,
        for later updating of the object by method adapt_set
        Input:
        --------
        a_state: Accumulated adaptation state, object which has field state_gen, output_distr, logprob
        obs_data: [n_samples, n_features] a sequence of data supposed to be drawn from this hmm
        
        Result:
        --------
        a_state: Accumulated adaptation state, including this subset of observed data.
        logP: Accumulated log( P(obs_data| hmm) )

        Method:
        --------
        From hmm.output_distr obtain observation probabilities. 
        hmm.state_gen uses output prob to compute conditional prob P(state | obs_data), 
        which are further used to adapt output_distr
        """

        pX, l_scale = self.output_distr[0].prob(obs_data, self.output_distr) # scaled observation prob
        # pX[i][t] * exp(l_scale[t]) == P(obs_data[t][:] | hmm.output_distr[i])
        a_state.MC, gamma, logP = self.state_gen.adapt_accum(a_state.MC, pX)
        # gamma[i][t] = P[hmmState = i | obs_data, hmm]
        a_state.Out = self.output_distr[0].adapt_accum(self.output_distr, a_state.Out, obs_data, gamma)
        if len(l_scale) == 1:
            # when? len(hmm.output_distr) == 1
            a_state.LogProb += logP + obs_data.shape[0] * l_scale
        else:
            a_state.LogProb += logP + np.sum(l_scale) # logprob(hmm, obs_data)
        return a_state, a_state.LogProb

    def adapt_set(self, a_state):
        """
        Set the HMM object using accumulated statistics from observed training data.
        Input:
        ------
        a_state: accumulated statistics from previous calls of adapt_accum
        """
        self.state_gen.adapt_set(a_state.MC)
        self.output_distr = self.output_distr[0].adapt_set(self.output_distr, a_state.Out)

    def rand(self, n_samples):
        """
        Genearte a random sequence of data from a given HMM.
        Input:
        ------
        n_samples: Maximum number of samples generated
        Return:
        ------
        X: [nS, n_features]. Sequence of generated output samples.
        S: [nS, ]. Sequence of integer state values
        """
        S = self.state_gen.rand(n_samples)
        nS = len(S)
        X = np.zeros((nS, self.data_size))
        for s in range(1, int(max(S)) + 1):
            X[S == s, :] = self.output_distr[s - 1].rand(sum(S == s)).reshape(-1, self.data_size)
        return X, S

    def viterbi(self, x):
        """
        Calculate optimal hmm state sequence given observation data sequence for a single hmm object.
        Input:
        ------
        x: [n_samples, n_features]. Sequence of observation data.
        Return:
        ------
        s_opt: [n_samples, ]. Optimal state sequence.
        logP: scalar. logP = log P(x, s_opt | hmm) 
        """
        T, n_features = x.shape
        s_opt = np.zeros((T))
        if n_features != self.data_size:
            raise ValueError("Observation data size must be consistent with hmm object")
        logp_x = self.output_distr[0].logprob(x, self.output_distr)
        s_opt, logP = self.state_gen.viterbi(logp_x)
        return s_opt, logP


class AState(object):
    def __init__(self, mc_state, out_state, logprob):
        self.MC = mc_state
        self.Out = out_state
        self.LogProb = logprob # to store the accumulated logprob of observation


def logprob(hmm, x):
    """
    logP = logprob(hmm,x) gives conditional log(probability densities)
    for an observed sequence of (possibly vector-valued) samples,
    for each HMM object in an array of HMM objects.
    Input:
    ------
    hmm: a single hmm object or a list of hmm objects
    x: observation data. [n_samples, n_features]
    Return:
    ------
    logP: 1D array. [n_objs, ]. logP[i] = log P[ x | hmm(i) ]
    Method:
    ------
    Run the forward algorithm with each hmm on the observation sequence.

    """
    if isinstance(hmm, HMM):
        hmm = [hmm]
    if isinstance(hmm, list) and isinstance(hmm[0], HMM):
        n_objs = len(hmm)
        n_samples, n_features = x.shape
        logP = np.zeros((n_objs))
        for i in range(0, n_objs):
            logp_act = 0
            pX, logS = hmm[i].output_distr[0].prob(x, hmm[i].output_distr)
            alpha_hat, c = hmm[i].state_gen.forward(pX)
            # compute true probability with scale factor
            if np.isscalar(logS):
                logS = np.tile(logS, (n_samples))
            for j in range(0, n_samples):
                logp_act += np.log(c[j]) + logS[j]
            if len(c) == n_samples:
                # ln(c_0) + .. + ln(c_{T-1})
                logP[i] = logp_act
            else:
                logP[i] = logp_act + np.log(c[-1]) # c[-1] is not scaled
    else:
        raise ValueError("The first input must be an hmm object or a list of hmm objects")
    return logP


def make_leftright_hmm(n_states, pD, obs_data, l_data=None):
    """
    Initialize and train a Hidden Markov Model to conform with a given set of training data sequence.
    Input:
    ------
    n_states: Desired number of HMM states.
    pD: a single object of some probability-distribution class
    obs_data: [n_samples, n_features]. The concatenated training sequences. One sample of observed data vector is stored row-wise.
    l_data: [n_sequence, ]. l_data[r] is the length of rth training sequence.
    Return:
    hmm: the trained left-right hmm object
    """
    if n_states <= 0:
        raise ValueError("Number of states must be >0")
    if l_data is None:
        l_data = [obs_data.shape[0]] # Just one single sequence
    # Make left-right Markov Chain with finite duration
    D = np.mean(l_data) / n_states # average state duration
    mc = MarkovChain()
    mc.init_left_right(n_states, D)
    hmm = HMM(mc, pD)
    hmm.init_leftright_outputdistr(obs_data, l_data) # crude initialize hmm.output_distr
    # standard training
    hmm.train(obs_data, l_data, 5, np.log(1.01))
    return hmm

