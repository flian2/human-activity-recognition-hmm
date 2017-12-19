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
        mean_s = np.array([[-3.0, 0.0],
                          [3.0, 0.0],
                          [0.0, -3.0],
                          [0.0, 3.0]])
        std_s = np.array([[1.0, 3.0],
                          [1.0, 3.0],
                          [3.0, 1.0],
                          [3.0, 1.0]])
        m_w = np.array([3.0, 3.0, 1.0, 1.0])
        gmm = make_gmm_single(mean_s, std_s, m_w)
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

    def test_init_by_data(self):
        mean_s = np.array([[-3.0, 0.0],
                          [3.0, 0.0],
                          [0.0, -3.0],
                          [0.0, 3.0]])
        std_s = np.array([[1.0, 3.0],
                          [1.0, 3.0],
                          [3.0, 1.0],
                          [3.0, 1.0]])
        m_w = np.array([3.0, 3.0, 1.0, 1.0])
        gmm = make_gmm_single(mean_s, std_s, m_w)

        # Generate training sequence
        np.random.seed(0)
        x, S = gmm.rand(10000)
        # Initialize a new gmm object using training data
        gmm_new = GaussMixDistr(4)
        gmm_new.init_by_data(x)
        for i in range(0, 4):
            print "Mixture %d" % i
            print "Mean: ", gmm_new.gaussians[i].mean
            print "Std: ", gmm_new.gaussians[i].std
        print "mix_weight: ", gmm_new.mix_weight
        colors = cm.rainbow(np.linspace(0, 1, n_mixtures))
        plot_samples_in_mixtures(x, S, colors)
        plot_mixture_centroids(gmm_new.gaussians, colors)

    def test_adapt_single_gmm(self):
        mean_s = np.array([[-3.0, 0.0],
                          [3.0, 0.0],
                          [0.0, -3.0],
                          [0.0, 3.0]])
        std_s = np.array([[0.5, 0.1],
                          [1.0, 0.1],
                          [0.5, 1.0],
                          [0.5, 1.0]])
        m_w = np.array([3.0, 3.0, 1.0, 1.0])
        gmm = make_gmm_single(mean_s, std_s, m_w)

        # Generate training sequence
        np.random.seed(0)
        x, S = gmm.rand(10000)
        # Initialize a new gmm object using training data
        # gmm_new = GaussMixDistr(4)
        # gmm_new.init_by_data(x)
        # for i in range(0, 4):
        #     print "Mixture %d" % i
        #     print "Mean: ", gmm_new.gaussians[i].mean
        #     print "Std: ", gmm_new.gaussians[i].std
        # print "mix_weight: ", gmm_new.mix_weight
        colors = cm.rainbow(np.linspace(0, 1, 4))
        mean_init = np.array([[-0.3, 0.0],
                      [0.3, 0.0],
                      [0.0, -0.3],
                      [0.0, 0.3]])
        std_init = np.ones((4, 2))
        gmm_new = make_gmm_single(mean_init, std_init)
        plot_samples_in_mixtures(x, S, colors)
        plot_mixture_centroids(gmm_new.gaussians, colors)

        # Train
        for n in range(0, 10):
            pD_list = [gmm_new]
            aS = gmm_new.adapt_start(pD_list)
            aS = gmm_new.adapt_accum(pD_list, aS, x)
            pD_list = gmm_new.adapt_set(pD_list, aS)
            plot_samples_in_mixtures(x, S, colors)
            plot_mixture_centroids(gmm_new.gaussians, colors)
        print "after training: mix_weight: ", gmm_new.mix_weight

def make_gmm_single(mean_s, std_s, m_w=None):
    gaussians = []
    n_mixtures = mean_s.shape[0]
    if m_w is None:
        m_w = np.ones((n_mixtures))
    for i in range(0, mean_s.shape[0]):
        gaussians.append( GaussDistr(mean=mean_s[i, :], std=std_s[i, :]) )
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

def plot_samples_in_mixtures(x, S, colors):
    n_mixtures = int(np.max(S)) + 1
    
    for i in range(0, n_mixtures):
        plt.scatter(x[S == i, 0], x[S == i, 1], c=colors[i])
    plt.legend(("mixture 1", "mixture 2", "mixture 3", "mixture 4"))

def plot_mixture_centroids(gaussians, colors):
    # Visualize mean and variance of gaussians on a 2D plot
    n_mixtures = len(gaussians)
    for i in range(0, n_mixtures):
        m = gaussians[i].mean
        v = np.dot(gaussians[i].cov_eigen, np.diag(gaussians[i].std))
        for k in range(0, v.shape[0]):
            x_start = m - v[:, k] # [n_features, 1]
            x_end = m + v[:, k]
            # show feature_0, feature[1]
            plt.plot([x_start[0], x_end[0]], [x_start[1], x_end[1]], c=colors[i])
    plt.show()


if __name__ == '__main__':
    unittest.main()