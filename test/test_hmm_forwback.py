import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
from hmm import HMM
from markov_chain import MarkovChain

def main():
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

if __name__ == "__main__":
    main()