import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
from hmm import HMM
from markov_chain import MarkovChain
import unittest

def main():
	q1 = np.array([1, 0])
	A1 = np.array([[0.9, 0.1, 0], [0, 0.9, 0.1]])
	mc = MarkovChain(q1, A1)
	pD_list = [DiscreteDistr(np.array([0.1, 0.2, 0.6])), DiscreteDistr(np.array([0.6, 0.35, 0.05]))]
	hmm1 = HMM(mc, pD_list)
	[X, S] = hmm1.rand(100)
	print "generated state sequence:"
	print S # check that the last state cannot be 1
	print "generated output sequence:"
	print X # check when state is 1: more likely output '2','3'. When state is 2: more likely output '1', '2'
	# logP1 = hmm1.logprob(X)


if __name__ == "__main__":
	main()