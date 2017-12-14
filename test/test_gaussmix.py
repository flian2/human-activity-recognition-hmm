import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
from gauss_distr import GaussDistr
from gaussmix_distr import GaussMixDistr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import unittest

class GaussMixDistrTest(unittest.TestCase):
    def test_rand(self):
        gaussians = []
        mean_s = np.array([[-3.0, 0.0],
                          [3.0, 0.0],
                          [0.0, -3.0],
                          [0.0, 3.0]])
        std_s = np.array([[1.0, 3.0],
                          [1.0, 3.0],
                          [3.0, 1.0],
                          [3.0, 1.0]])
        for i in range(0, 4):
            gaussians.append( GaussDistr(mean=mean_s[i, :], std=std_s[i, :]) )
        m_w = np.array([3.0, 3.0, 1.0, 1.0])
        gmm = GaussMixDistr(gaussians, m_w)
        x, S = gmm.rand(10000)

        # plot random samples
        n_mixtures = int(np.max(S)) + 1
        colors = cm.rainbow(np.linspace(0, 1, n_mixtures))
        
        for i in range(0, n_mixtures):
            plt.scatter(x[S == i, 0], x[S == i, 1], c=colors[i])
        plt.legend(("mixture 1", "mixture 2", "mixture 3", "mixture 4"))
        # should be four mixtures
        plt.show()

    def test_logprob(self):
        gmm = make_gmm_single()
        gmm_list = make_gmm_two()

        x = np.array([[-3.0, 0.0], 
                      [3.0, 0.0],
                      [0.0, -3.0],
                      [0.0, 3.0],
                      [-1.5, 2.5],
                      [-1.0, -2.0],
                      [4.5, -10.0],
                      [7.5, 2.5]])

        logP1 = gmm.logprob(x)
        logP2 = gmm_list[0].logprob(x, gmm_list)

        logP1_expected = np.array([[-3.9128, -3.9128, -4.9763, -4.9763, -4.6326, -5.1217, -10.5979, -8.2637]])
        logP2_expected = np.array([[-3.9128, -3.9128, -4.9763, -4.9763, -4.6326, -5.1217, -10.5979, -8.2637],
                                   [-4.4769, 1.0933, -3.2904, -2.9798, -5.2248, -4.9326, -21.0586, -29.2320]])
        np.testing.assert_array_almost_equal(logP1, logP1_expected, decimal=4)
        np.testing.assert_array_almost_equal(logP2, logP2_expected, decimal=4)

def make_gmm_single():
    gaussians = []
    mean_s = np.array([[-3.0, 0.0],
                      [3.0, 0.0],
                      [0.0, -3.0],
                      [0.0, 3.0]])
    std_s = np.array([[1.0, 3.0],
                      [1.0, 3.0],
                      [3.0, 1.0],
                      [3.0, 1.0]])
    for i in range(0, 4):
        gaussians.append( GaussDistr(mean=mean_s[i, :], std=std_s[i, :]) )
    m_w = np.array([3.0, 3.0, 1.0, 1.0])
    gmm = GaussMixDistr(gaussians, m_w)
    return gmm

def make_gmm_two():
    # generate two gmms with same mean and different covariance matrix.
    gaussians1 = []
    gaussians2 = []
    mean_s = np.array([[-3.0, 0.0],
                      [3.0, 0.0],
                      [0.0, -3.0],
                      [0.0, 3.0]])
    std_s1 = np.array([[1.0, 3.0],
                      [1.0, 3.0],
                      [3.0, 1.0],
                      [3.0, 1.0]])
    std_s2 = np.array([[1.5, 3.5],
                      [0.1, 0.2],
                      [0.5, 1.1],
                      [0.4, 1.0]])
    for i in range(0, 4):
        gaussians1.append( GaussDistr(mean=mean_s[i, :], std=std_s1[i, :]) )
        gaussians2.append( GaussDistr(mean=mean_s[i, :], std=std_s2[i, :]) )

    m_w = np.array([3.0, 3.0, 1.0, 1.0])
    gmm1 = GaussMixDistr(gaussians1, m_w)
    gmm2 = GaussMixDistr(gaussians2, m_w)
    gmm_list = [gmm1, gmm2]
    return gmm_list


if __name__ == '__main__':
    unittest.main()