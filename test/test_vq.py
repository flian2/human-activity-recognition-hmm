import sys
sys.path.insert(0, '../')
import numpy as np
from discrete_distr import DiscreteDistr
from gauss_distr import GaussDistr
from gaussmix_distr import GaussMixDistr
from vector_quantizer import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import Voronoi, voronoi_plot_2d
import unittest
import timeit

def main():
    np.random.seed(0)
    plot_vq_result(100)

def plot_vq_result(n_codes):
    # generate random data and train vector quantizer on the data. 
    # Show the voronoi graph
    x = np.random.normal(0, 1, (50000, 2))
    start_time = timeit.default_timer()
    vq, _ = make_VQ(x, n_codes)
    elapsed = timeit.default_timer() - start_time
    print "training time: ", elapsed
    cb = vq.code_book
    vor = Voronoi(cb)

    voronoi_plot_2d(vor)
    plt.scatter(x[:, 0], x[:, 1], 0.2, alpha=0.1)

    plt.show()

if __name__ == '__main__':
    main()
