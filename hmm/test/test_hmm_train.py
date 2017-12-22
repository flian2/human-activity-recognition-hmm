import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from discrete_distr import DiscreteDistr
from gauss_distr import GaussDistr
from gaussmix_distr import GaussMixDistr
from hmm import *
from markov_chain import MarkovChain
from test_gaussmix import plot_samples_in_mixtures
import unittest

class HmmTrainTest(unittest.TestCase):
    def test_train_leftrighthmm_gaussd(self):
        n_states = 3
        A = np.array([[0.95, 0.05, 0.0, 0.0],
                     [0.0, 0.95, 0.05, 0.0],
                     [0.0, 0.0, 0.95, 0.05]]) 
        p0 = np.array([1, 0, 0])
        mc = MarkovChain(p0, A)
        pD = []
        pD.append( GaussDistr(mean=np.array([0.0, 0.0]), std=np.array([3.0, 1.0])) )
        pD.append( GaussDistr(mean=np.array([1.0, 0.0]), std=np.array([1.0, 3.0])) )
        pD.append( GaussDistr(mean=np.array([-1.0, 0.0]), std=np.array([1.0, 3.0])) )
        # Generate training samples
        hmm = HMM(mc, pD)
        x_training = np.empty((0, 2))
        l_xT = np.empty((0)) # length of subsequence
        sT = np.empty((0)) # state sequence
        np.random.seed(100)
        for t in range(0, 20):
            x, s = hmm.rand(1000)
            x_training = np.concatenate((x_training, x), axis=0)
            sT = np.concatenate((sT, s))
            l_xT = np.concatenate( (l_xT, np.array([s.shape[0]])) )
        # print "length of subsequence: ", l_xT

        # Initialize new hmm, only know n_states = 3, distribution is Gaussian.
        hmm_new = init_leftright_hmm(3, GaussDistr(), x_training, l_xT)
        # Train hmm, simplify procedure in hmm.train
        i_xT = np.append(np.array([0]), np.cumsum(l_xT)) # start index for each subsequence
        for n_training in range(0, 10):
            aS = hmm_new.adapt_start()
            for r in range(0, len(l_xT)):
                aS, _ = hmm_new.adapt_accum(aS, x_training[i_xT[r]: i_xT[r+1], :])
            hmm_new.adapt_set(aS)

        # Verify hmm_new is close enough to the original hmm
        np.testing.assert_allclose(hmm_new.state_gen.initial_prob, p0, rtol=0.01)
        np.testing.assert_allclose(hmm_new.state_gen.transition_prob, A, rtol=0.2)
        
        for i in range(0, 3):
            print "true mean for state %d: " % i, hmm.output_distr[i].mean
            print "estimated mean for state %d: " % i, hmm_new.output_distr[i].mean
            print "true std for state %d: " % i, hmm.output_distr[i].std
            print "estimated std for state %d: " % i, hmm_new.output_distr[i].std

    def test_leftrighthmm_gmm(self):
        n_states = 3
        A = np.array([[0.99, 0.01, 0.0, 0.0],
                      [0.0, 0.99, 0.01, 0.0],
                      [0.0, 0.0, 0.99, 0.01]])
        p0 = np.array([1.0, 0.0, 0.0])
        mc = MarkovChain(p0, A)

        mean_init = np.array([[0.0, 0.0],
                              [1.0, 0.0],
                              [-1.0, 0.0]])
        std_init = np.array([[3, 1],
                             [3, 1],
                             [1, 3]])
        gaussD_list = []
        for i in range(0, 3):
            gaussD_list.append(GaussDistr(mean=mean_init[i, :], std=std_init[i, :]))
        hmm_gen = HMM(mc, gaussD_list)

        # Prepare for training
        x_training = np.empty((0, 2))
        l_xT = np.empty((0)) # length of subsequence
        sT = np.empty((0)) # state sequence
        np.random.seed(100)
        # generate 20 training subsequences
        for t in range(0, 20):
            x, s = hmm_gen.rand(1000)
            x_training = np.concatenate((x_training, x), axis=0)
            sT = np.concatenate((sT, s))
            l_xT = np.concatenate( (l_xT, np.array([s.shape[0]])) )

        hmm_new = init_leftright_hmm(n_states, GaussMixDistr(2), x_training, l_xT)
        # Train hmm
        i_xT = np.append(np.array([0]), np.cumsum(l_xT)) # start index for each subsequence
        colors = cm.rainbow(np.linspace(0, 1, 4))
        logP_per_training = []
        for n_training in range(0, 15):
            plot_samples_in_mixtures(x_training, sT, colors)
            gmms = hmm_new.output_distr
            gmms[0].plot_mixture_centroids(gmms, colors)
            plt.show()

            aS = hmm_new.adapt_start()
            for r in range(0, len(l_xT)):
                aS, logP = hmm_new.adapt_accum(aS, x_training[i_xT[r]: i_xT[r+1], :])
            hmm_new.adapt_set(aS)
            logP_per_training.append(logP)


def init_leftright_hmm(n_states, pD, obs_data, l_data=None):
    """
    Initialize a Hidden Markov Model to conform with a given set of training data sequence. Skip the train step.
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
    return hmm

if __name__ == "__main__":
    unittest.main()