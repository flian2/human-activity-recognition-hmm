import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
import unittest

class DiscreteDistrTest(unittest.TestCase):

    def test_single_discreteD(self):
        dD = DiscreteDistr(np.array([3, 3, 1, 1]))
        x_samples = dD.rand(50)
        print "generated samples: "
        print x_samples # expects to see more 1, 2 than 3, 4
        p, logS = dD.prob(x_samples)
        print "sample probabilities:"
        print p
        self.assertEqual(logS, 0)

        # use x_samples to init a DiscreteDistr object
        x_samples_large = dD.rand(1000)
        dD2 = DiscreteDistr()
        dD2.init_by_data(x_samples_large)
        print dD2.prob_mass

    def test_adapt_single_discreteD(self):
        true_prob_mass = np.array([3, 3, 0, 1, 0])
        dD = DiscreteDistr(true_prob_mass)
        # large training data
        np.random.seed(10)
        x_samples = dD.rand(3000)
        # use samples to train a new discrete distributino
        dD_train = DiscreteDistr(np.ones((5)))
        dD_train.pseudo_count = 1 # smooth
        aS = dD_train.adapt_start()
        aS = dD_train.adapt_accum(aS, x_samples)
        dD_train.adapt_set(aS)
        train_prob_mass = dD_train.prob_mass

        np.testing.assert_array_almost_equal(train_prob_mass, \
            true_prob_mass * 1.0 / np.sum(true_prob_mass) , decimal=2)


if __name__ == "__main__":
    unittest.main()