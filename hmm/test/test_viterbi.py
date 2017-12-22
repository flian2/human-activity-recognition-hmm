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

class HmmViterbiTest(unittest.TestCase):
	def test_hmm_discrete(self):
		# Example from http://www.cis.upenn.edu/~cis262/notes/Example-Viterbi-DNA.pdf  
		p0 = np.array([0.5, 0.5]) # states: H, L
		#     H    L
		# H 0.5   0.5
		# L 0.4   0.6
		A = np.array([[0.5, 0.5], [0.4, 0.6]])
		mc = MarkovChain(p0, A)
		pd_H = DiscreteDistr(np.array([0.2, 0.3, 0.3, 0.2])) # probability for A, C, G, T
		pd_L = DiscreteDistr(np.array([0.3, 0.2, 0.2, 0.3]))
		hmm = HMM(mc, [pd_H, pd_L])

		# Output sequence GGCACTGAA
		obs_data = np.array([3, 3, 2, 1, 2, 4, 3, 1, 1])[:, np.newaxis]
		# expect path HHHLLLLLL
		s_exp = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2])
		s_opt, logP = hmm.viterbi(obs_data)
		print logP
		np.testing.assert_array_equal(s_opt, s_exp)

if __name__ == '__main__':
	unittest.main()
