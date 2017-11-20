import numpy as np
import copy
class MarkovChain:
    """
    Markov Chain class
    n_states: number of states in the Markov Chain
    initial_prob: Initial probablity distribution [n_states, ]
    transitional_prob: Transition probability matrix [n_states, n_states]
    """
    def __init__(self, q=np.array(1.0), A=np.array(1.0)):
        self.initial_prob = q
        self.transition_prob = A

    @property
    def n_states(self):
        return self.initial_prob.shape[0]


class HMM:
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
        d_size = pD[0].data_size
        if all( pD[i].data_size == d_size for i in range(0, len(pD)) ):
            self._output_distr = pD
        else:
            raise ValueError("All distribution must have the same data size")

    def init_leftright_hmm(self, obs_data, len_seq):
        """
        Initionlize output distribution crudely according to observation data for left-hmm
        obs_data: [n_samples, n_features]. The concatenated training sequences.
        len_seq: [n_sequence, ]. Length of subsequences of the training data. sum(len_seq) = n_samples
        """
        # Check input size
        if len_seq.sum() != obs_data.shape[0]:
            raise ValueError("Training data has %d samples, expecing the same number in len_seq" % obs_data.shape[0])
        # Make output_distr a vector of the same size of hmm states
        if not isinstance(self.output_distr, list):
            self.output_distr = [copy.deepcopy(self.output_distr) for i in range(0, self.n_states)]

        n_samples, n_features = obs_data.shape
        n_sequences = len(len_seq)
        start_ind = np.array([0]).append(np.cumsum(len_seq))
        n_states = self.n_states
        # Use average length length of each state to initialize output distribution
        for i in range(0, len(n_states)):
            data_per_state = np.zeros((0, n_features))
            for r in range(0: len(n_sequences)):
                # staring point for the i-th state in the r-th subsequence
                d_start = start_ind[r] + (i * len_seq[r]) / n_states
                d_end = start_ind[r] + ((i+1) * len_seq[r]) / n_states # exclusive
                data_per_state = np.concatenate((data_per_state, obs_data[d_start:d_end, :]), axis=0)
            # Very crude initialization, should be refined by training
            # to do: can we skip init here? Just train later
            self.output_distr[i] = self.output_distr.init(data_per_state) 
