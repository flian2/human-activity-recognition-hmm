import numpy as np
from scipy.sparse import csr_matrix
from discrete_distr import DiscreteDistr

# Based on the MATLAB code from 
# A. Leijon, "Pattern recognition fundamental theory and exercise problems," KTH Elec- tral Engineering, 2010
class MarkovChain(object):
    """
    Markov Chain class
    n_states: number of states in the Markov Chain
    initial_prob: Initial probablity distribution [n_states, ]
    transition_prob: Transition probability matrix [n_states, n_states]
    """
    def __init__(self, q=np.array(1.0), A=np.array(1.0)):
        self.initial_prob = q
        self.transition_prob = A

    @property
    def n_states(self):
        return self.transition_prob.shape[0]

    def init_left_right(self, n_states, state_duration=10):
        """
        Initialize a single MarkovChain object to a finite-duration first-order left-right structure
        allowing transitions from every state only to the nearest following state.
        Inputs:
        -------
        n_states: Desired number of Markov states
        state_duration: Average number of consecutive samples in each state.
                        len(state_duration) == n_states or state_duration is scalar.
        """
        if n_states < 1:
            raise ValueError("Number of states must be < 1")
        if np.isscalar(state_duration):
            state_duration = [state_duration] * n_states
        elif len(state_duration) != n_states:
            raise ValueError("Length of state_duration must be equal to n_states")

        min_diag_prob = 0.1 # Initial min diagonal transition prob value.
        D = np.array([max(1, dur) for dur in state_duration], dtype='float')
        # Diagonal values of transition prob matrix
        aii = np.array([max(min_diag_prob, val) for val in (D - 1) / D], dtype='float')
        # Off diagonal values, only one non-zero off-diagonal
        aij = 1 - aii;
        # to do: make A matrix sparse
        A = np.zeros((n_states, n_states + 1)) # has an end state, thus (n_states + 1) column
        for i in range(0, n_states):
            A[i][i] = aii[i]
            A[i][i+1] = aij[i]
        p0 = np.concatenate( (np.ones((1)), np.zeros((n_states - 1))) )
        self.initial_prob = p0
        self.transition_prob = A

    def adapt_start(self):
        """
        Initialize adaptation data structure.
        Return:
        aS: Zero initialized adaptation data object.

        """
        aS = McAState()
        aS.pI = np.zeros(self.initial_prob.shape) # accumulated sum of pI[j] == P[ s(1) = j | each training sub-sequence]
        aS.pS = np.zeros(self.transition_prob.shape) # accumulated sum of pS[i][j] == P[ s(t)=i, s(t+1)=j | all training data]
        return aS

    def adapt_accum(self, a_state, pX):
        """
        Adapt a single MarkovChain object to observed data, by accumulating sufficient statistics from the data.
        Input:
        ------- 
        a_state: accumulated adapation state from previous calls
        pX: pX[j][t] = scalefactor *  P[ X(t) = obs_data[t][:] | S(t) = j ]
        Return:
        -------
        a_state: accumulated adapation state including this step.
        a_state.pI: 1D array. Accumulated sum of pI[j] == P[ s(1) = j | each training sub-sequence]
        a_state.pS: 2D array. Accumulated sum over t of pS[i][j] == P[ s(t)=i, s(t+1)=j | all training data]
        gamma: gamma[i][t]=P[ S(t)= i | pX for complete observation sequence]
        logP: scalar log(Prob(observed sequence))

        Method:
        Results of forward-backward algorithm combined with Baum-Welch update rule.
        """
        T = pX.shape[1] # length of sequence
        n_states = self.n_states
        A = self.transition_prob

        # Get scaled forward backward variables
        alpha_hat, c = self.forward(pX)
        beta_hat = self.backward(pX, c)
        
        # Compute gamma
        gamma = np.multiply( np.multiply(alpha_hat, beta_hat), np.tile(c[:T], (n_states, 1)) )
        # Initial probabilities, a_state.pI += gamma(t=0)
        a_state.pI += gamma[:, 0]

        # Calculate xi for the current sequence
        # xi(i, j, t) = alpha_hat(i, t) * A(i, j) * pX(j, t+1) * beta_hat(j, t+1)

        # Elementwise multiply pX .* beta_hat
        pXbH = np.multiply(pX[:, 1:], beta_hat[:, 1:])
        # alpha_hat(i, t) * pX(j, t+1) * beta_hat(j, t+1) sum over t = 0... T-2
        aHpXbH = alpha_hat[:, : T - 1].dot(pXbH.T)
        # [n_states, n_states] array where (i,j) element is xi(i, j, t) summed over t = 0...T-2
        xi = np.multiply(A[:, :n_states], aHpXbH)
        # Add xi to pS
        a_state.pS[:, :n_states] += xi

        if A.shape[0] != A.shape[1]:
            # finite-duration hmm
            a_state.pS[:, n_states] += alpha_hat[:, T - 1] * beta_hat[:, T - 1] * c[T - 1]
        # log(P(current sequence | hmm))
        logP = np.sum(np.log(c))

        return a_state, gamma, logP

    def adapt_set(self, a_state):
        """
        Final adapation for a MarkovChain object using accumulated statistics.
        Input:
        ------
        a_state: accumulated adapation state from previous calls of adapt_accum
        """
        self.initial_prob = a_state.pI / np.sum(a_state.pI) # normalized
        self.transition_prob = np.divide( a_state.pS, \
            np.tile( np.sum(a_state.pS, axis=1)[:, np.newaxis], (1, a_state.pS.shape[1]) ) )
        # if state i is never encountered in training, set trainsition prob to A[i,i]=1, A[i,j]=0
        tmp = np.eye((self.n_states))
        nan_rows = np.any(np.isnan(self.transition_prob), axis=1)
        tmp[np.logical_not(nan_rows), :] = 0
        self.transition_prob[nan_rows, :self.n_states] = tmp[nan_rows, :]

    def forward(self, pX):
        """
        Calculate state and observation probabilities for a single data sequence, using the forward algorithm, 
        given a single MarkovChain object.
        If hmm is finite duration, the state reaches the end state after the last sequence, S(T) = N + 1.
        Input:
        -------
        pX: [n_states, T]
        pX[j, t] = scalefactor(t) *  P[ X(t) = obs_data[t][:] | S(t) = j ], for j = 0...N-1, t = 0...T-1.

        Return:
        -------
        alpha_hat: [n_states, T] matrix with normalized state probabilities, given the observations
                   alpha_hat[j][t] = P[ S(t) = j | x(1),...x(t), HMM ]
        c: 1D array with observation probabilities given the HMM
           c[t] = P[x(t) | x(0), ... x(t-1), HMM ] t = 0,...,T-1
           c[0]*c[1]*..c[t] = P[ x[1]..x[t]| HMM ]
           If HMM is finite duration, c[T]= P[ S[T]=N+1 | x[0]...x[T-1], HMM ]
        Thus, for inifinite duration HMM:


        for infinite-duration HMM:
            len(c) == T, prod(c) = P(x[0], x[1], ... x[T-1])
        for finite-duration HMM:
            len(c) = T+1, prod(c) =  P(x[0], x[1], ... x[T-1], x[T]=END)
        """
        T = pX.shape[1]
        n_states = self.n_states
        q = self.initial_prob # 1D array
        A = self.transition_prob
        B = pX
        rows, columns = A.shape
        if rows != columns:
            c = np.empty(shape=(T + 1)) # 1D array
        else:
            c = np.empty(shape=(T))
        alpha_hat = np.zeros(pX.shape)

        # Initialize init_alpha_tmp, c, alpha_hat
        init_alpha_tmp = q * B[:, 0] # t = 0
        c[0] = np.sum(init_alpha_tmp)
        alpha_hat[:, 0] = init_alpha_tmp / c[0] # initialize first column of alpha_hat

        for t in range(1, T):
            alpha_tmp = np.empty((n_states)) # 1D
            for j in range(0, n_states):
                alpha_tmp[j] = B[j, t] * np.inner( alpha_hat[:, t-1], A[:, j] )
            c[t] = np.sum(alpha_tmp)
            alpha_tmp /= c[t]
            # if c[t] == 0:
            #     print alpha_tmp
            alpha_hat[:, t] = alpha_tmp

        if rows != columns:
            c[T] = np.inner(alpha_hat[:, T - 1], A[:, columns - 1])

        # if not np.any(np.isnan(pX)) and np.any(np.isnan(alpha_hat)):
        #     print "q"
        #     print q
        #     print "alpha_tmp"
        #     print alpha_tmp
        #     print "alpha_hat"
        #     print alpha_hat
        #     print "transition_prob"
        #     print self.transition_prob

        return alpha_hat, c

    def backward(self, pX, c):
        """
        Calculate scaled observation probabilities, using backward algorithm, for a given MarkovChain.

        Input:
        ------
        pX: [n_states, T]
        pX[j, t] = scalefactor(t) *  P[ X(t) = obs_data[t][:] | S(t) = j ], for j = 0...N-1, t = 0...T-1.
        c: 1D array with observation probabilities given the HMM
           c[t] = P[x(t) | x(0), ... x(t-1), HMM ] t = 0,...,T-1

        Return:
        beta_hat: [n_states, T] matrix with scaled backward probabilities.
                 beta_hat[j, t] = beta[j, t] / (c[0],...c[T-1]), where
                 beta[j, t] = P[ x(t+1)...x(T-1) | S(t)=j ], for infinite-duration HMM
                 beta[j, t] = P[ x(t+1)...x(T-1), x(T)=END | S(t)=j ] for finite-duration HMM.

        Note:
        For an infinite-duration HMM:
            P[ S(t)=j | x(0),...,x(T-1) ] = alpha_hat[j, t] * beta_hat[j, t] * c(t)
        For a finite-duration HMM with a separate END state:
            P[ S(t)=j | x(0),...,x(T-1),x(T)=END ] = alpha_hat[j, t] * beta_hat[j, t] * c(t)

        """
        T = pX.shape[1]
        n_states = self.n_states
        beta_hat = np.empty((n_states, T))
        beta = np.empty((n_states, T))
        # Initialize t = T-1
        if self.transition_prob.shape[1] == n_states:
            # infinite duration
            finite = False
            beta[:, T - 1] = np.ones((n_states))
            beta_hat[:, T - 1] = np.ones((n_states)) / c[T - 1]
        elif self.transition_prob.shape[1] == n_states + 1:
            finite = True
            beta[:, T - 1] = self.transition_prob[:, n_states]
            beta_hat[:, T - 1] = beta[:, T - 1] / (c[T - 1] * c[T])

        for t in range(T - 2, -1, -1):
            b_beta = pX[:, t + 1] * beta_hat[:, t + 1] # [n_states, ]
            beta_hat[:, None, t] = self.transition_prob[:, :n_states].dot(b_beta[:, np.newaxis]) / c[t]

        return beta_hat

    def rand(self, n_samples):
        """
        Generate a random state sequence of length n_samples.
        Input:
        ------
        n_samples: Maximum length of generated sequence.
        Return:
        ------
        S: [n_samples, ]. Generated integer state sequence, not including the end state.
            When mc is finite-duration, len(S) < n_samples; when mc is infinite-duration, len(S) == n_samples.
            S[i] can take on value 1~n_states
        """
        S = np.zeros((n_samples))
        n_states = self.n_states
        init_state = DiscreteDistr(self.initial_prob)
        state_t = DiscreteDistr(self.initial_prob)
        for i in range(0, n_samples):
            # Generate one sample using the distribution of current state
            S[i] = state_t.rand(1)
            if S[i] == n_states + 1:
                return S[:i] # do not include the END state
            p_mass = self.transition_prob[S[i] - 1, :]
            state_t = DiscreteDistr(p_mass)
        return S

    def viterbi(self, logp_x, allowZeroProb=False):
        """
        Viterbi algorithm for calculating the optimal state sequence in a Markov Chain,
        given log P(obs(t) | S(t) = i).
        Input:
        ------
        logp_x: [n_states, n_samples]. logp_x[i, t] = log P(obs(t) | S(t) = i)
        Return:
        ------
        s_opt: [n_samples, ]. Vector with most probable state sequence.
        logP: log P(x(0)...x(T-1), S(0),...S(T-1) | hmm )
        Method:
        ------
        Optimal state sequence maximize the probability: P(x(0),...,x(T-1), S(0),...,S(T-1) | hmm)
        """
        T = logp_x.shape[1]
        n_states = self.n_states

        # prevent probability to be zero
        if allowZeroProb is True:
            log_A = np.log(self.transition_prob)
        else:
            log_A = np.log(np.maximum(self.transition_prob, np.finfo(float).tiny)) # convert zero probability to realmin
            # log_A = np.log(self.transition_prob)
            # log_A[np.isinf(log_A)] = np.log(np.finfo(float).tiny) * 10 
            # # make this number small than log(realmin) in case of low output distr likelihood

        # log of transition prob matrix, prevents -inf
        log_A = log_A[:, :n_states]
        back_pointer = np.zeros((n_states, T))
        s_opt = np.zeros((T))

        # delta = log P(x(0), s(0) | hmm)
        if allowZeroProb is True:
            delta = np.log(self.initial_prob) + logp_x[:, 0]
        else:
            delta = np.log(np.maximum(self.initial_prob, np.finfo(float).tiny)) + logp_x[:, 0]
        for t in range(1, T):
            # i_delta[j] = argmax_i {log P ( s(t-1)=i, s(t)=j, x(0)... x(t-1) | hmm ) }
            p_ij = np.tile(delta[:, np.newaxis], (1, n_states)) + log_A
            i_delta = np.argmax(p_ij, axis=0)
            delta = p_ij[i_delta, np.arange(0, n_states)] # 1D array
            # delta[j] = max_{i_0, ... i_{t-1}}{log P(s(0)=i_0, .. s(t)=i_{t}, x(0),...x(t) | hmm)}
            delta += logp_x[:, t]
            back_pointer[:, t] = i_delta
        # delta[j] = max_{i_0, ... i_{T-2}}{log P(s(0)=i_0, .. s(T)=i_{T-1}, x(0),...x(T-1) | hmm)}
        # Backtrack
        i_delta = np.argmax(delta)
        logP = delta[i_delta]
        s_opt[T - 1] = i_delta
        for t in range(T - 2, -1, -1):
            s_opt[t] = back_pointer[s_opt[t + 1], t + 1]

        return s_opt + 1, logP # state index from 1


class McAState(object):
    """
    adapation data structure, has field pI (initial probability), pS (state transitional probability) 
    """
    def __init__(self):
        self.pI = None
        self.pS = None