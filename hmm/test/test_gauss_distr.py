import sys
sys.path.insert(0, '../')
import numpy as np
from gauss_distr import GaussDistr
import unittest
from matplotlib import pyplot as plt

class GaussDistrTest(unittest.TestCase):
    def test_rand(self):
        # random variable, 1 dimension
        gD1 = GaussDistr(mean=np.array([2.0]), std=np.array([0.2]))
        x_training = gD1.rand(1000)
        plt.hist(x_training[:, 0], bins=20)
        plt.show()
        print "sample mean: %.3f. sample var: %.3f" % (np.mean(x_training, axis=0), np.var(x_training, axis=0))

        # random vector, 2 dimension
        gD2 = GaussDistr(mean=np.array([2.0, -2.0]), std=np.array([1, 0.1]))
        x_training = gD2.rand(2000)
        plt.scatter(x_training[:, 0], x_training[:, 1], alpha=0.2)
        plt.xlim([-5, 5])
        plt.ylim([-4, 0])
        plt.show()

    def test_logprob(self):
        # One dimension Gaussian distribution
        gD1 = GaussDistr(mean=np.array([2.0]), std=np.array([0.2]))
        x = np.array([0, 0.5, 1, 2, 2.5, 3])[:, np.newaxis]
        logP_actual = gD1.logprob(x)
        logP_exp = np.array([[-49.3095,  -27.4345,  -11.8095, 0.6905, -2.4345, -11.8095]])
        np.testing.assert_array_almost_equal(logP_actual, logP_exp)

        # Multi dimensional, uncorrelated
        gD2 = GaussDistr(mean=np.array([-3.0, -1.0, 0.0]), std=np.array([1.0, 3.0, 2.0]))
        np.random.seed(0)
        x = np.array([[-3.0, -1.1, 0.2],
                     [-2.5, -1.0, -0.1],
                     [2.0, -0.2, 0.0],
                     [0.0, 0.0, 0.0]])
        logP_actual = gD2.logprob(x)
        logP_exp = np.array([[-4.5541, -4.6748, -17.0841, -9.1041]])
        np.testing.assert_array_almost_equal(logP_actual, logP_exp, decimal=4)

        # Multi dimensional, correlated
        cov = np.array([[1.0, -1.0], [-1.0, 3.0]])
        gD3 = GaussDistr(mean=np.array([-1.0, 1.0]), covariance=cov)
        x = np.array([[1.1, -2.0],
                     [0.0, -1.1],
                     [1.0, -1.0],
                     [3.0, -5.0]])
        logP_actual = gD3.logprob(x)
        logP_exp = np.array([[-4.591951, -2.986951, -4.184451, -11.184451]])
        np.testing.assert_array_almost_equal(logP_actual, logP_exp)

    def test_adapt_single_GaussD(self):
        true_mean = np.array([-3.0, -1.0])
        true_std = np.array([1.0, 3.0])
        gD = GaussDistr(mean=true_mean, std=true_std)
        train_distr = [gD]
        x_training = gD.rand(1000)

        aS = gD.adapt_start(train_distr)
        aS = gD.adapt_accum(train_distr, aS, x_training)
        train_distr = gD.adapt_set(train_distr, aS)

        esti_mean = train_distr[0].mean
        esti_std = train_distr[0].std
        np.testing.assert_allclose(esti_mean, true_mean, rtol=0.1)
        np.testing.assert_allclose(esti_std, true_std, rtol=0.1)

    def test_init_by_data(self):
        gD = GaussDistr(mean=np.array([1.0, -1.0]), std=np.array([0.5, 2.0]))
        x_training = gD.rand(1000)

        gD_new = GaussDistr()
        gD_new.init_by_data(x_training)
        
        np.testing.assert_allclose(gD_new.mean, np.array([1.0, -1.0]), rtol=0.1)
        np.testing.assert_allclose(gD_new.std, np.array([0.5, 2.0]), rtol=0.1)
        self.assertEqual(gD_new.allow_corr, False)


if __name__ == '__main__':
    unittest.main()