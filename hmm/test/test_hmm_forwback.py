import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
from gauss_distr import GaussDistr
from hmm import *
from markov_chain import MarkovChain
import unittest


class HmmForwBackTest(unittest.TestCase):
    def test_discrete_outdistr(self):
        q1 = np.array([1, 0])
        A1 = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
        mc = MarkovChain(q1, A1)
        pD_list = [DiscreteDistr(np.array([0.6, 0.3, 0.1])), DiscreteDistr(np.array([0.1, 0.3, 0.6]))]
        hmm1 = HMM(mc, pD_list)
        n_states = hmm1.n_states
        Z = np.array([1, 3, 2])
        T = len(Z)
        pZ, _ = pD_list[0].prob(Z, pD_list)

        [alpha_hat, c] = mc.forward(pZ)
        expected_alpha_hat = np.array([[1.0000, 0.6000, 0.5625],
                                      [0,    0.4000,  0.4375]])
        np.testing.assert_array_almost_equal(alpha_hat, expected_alpha_hat, decimal=4)

        beta_hat = mc.backward(pZ, c)
        expected_beta_hat = np.array([[ 1.6667,    1.5873,         0], 
                                     [12.8571,   14.2857,    7.9365]])
        np.testing.assert_array_almost_equal(beta_hat, expected_beta_hat, decimal=4)

        gamma = np.multiply(np.multiply(alpha_hat, beta_hat), np.tile(c[0:T], (n_states,1))) # to check
        expected_gamma =  np.array([[1.0000,    0.1429,    0],
                                   [0,    0.8571,    1.0000]])
        np.testing.assert_array_almost_equal(gamma, expected_gamma, decimal=4)

    def test_gauss_outdistr(self):
        p0 = np.array([1, 0])
        A = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
        mc = MarkovChain(p0, A)
        pD_list = []
        pD_list.append(GaussDistr(mean=np.array([0]), std=np.array([1])))
        pD_list.append(GaussDistr(mean=np.array([3]), std=np.array([2])))
        h = HMM(mc, pD_list)
        n_states = h.n_states
        x = np.array([-0.2, 2.6, 1.3])[:, np.newaxis]
        T = x.shape[0]

        pX, logS = pD_list[0].prob(x, pD_list)
        alpha_hat, c = mc.forward(pX)
        beta_hat = mc.backward(pX, c)
        logP_hmm = logprob(h, x)
        
        pX_exp = np.array([[1.0000, 0.0695, 1.0000],
                           [0.1418, 1.0000, 0.8111]])
        np.testing.assert_array_almost_equal(pX, pX_exp, decimal=4)

        alpha_hat_exp = np.array([[1.0000, 0.3847, 0.4189], 
                                  [0, 0.6153, 0.5811]])
        np.testing.assert_array_almost_equal(alpha_hat, alpha_hat_exp, decimal=4)

        c_exp = np.array([1.0000, 0.1625, 0.8266, 0.0581])
        np.testing.assert_array_almost_equal(c, c_exp, decimal=4)

        beta_hat_exp = np.array([[1.0000, 1.0389, 0],
                                 [8.4154, 9.3504, 2.0818]])
        np.testing.assert_array_almost_equal(beta_hat, beta_hat_exp, decimal=4)

        logP_hmm_exp = np.array([-9.1877])
        np.testing.assert_array_almost_equal(logP_hmm, logP_hmm_exp, decimal=4)


if __name__ == "__main__":
    unittest.main()