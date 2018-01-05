import numpy as np
import warnings

# Based on the MATLAB code from 
# A. Leijon, "Pattern recognition fundamental theory and exercise problems," KTH Elec- tral Engineering, 2010
class VectorQuantizer(object):
    """
    Vector quantizer class. 
    Properties:
    -----------
    code_book: [n_codes, n_features]. Set of vector centroids used for quantization, stored row-wise
    """

    def __init__(self, cb=np.array([])):
        self.code_book = cb

    def init_by_data(self, x, n_codes):
        """
        Initialize the VectorQuantizer 
        Input:
        ------
        x: training data [n_samples, n_features].
        n_codes: desired size of code book, must satisfy n_codes <= n_samples
        """
        n_samples, n_features = x.shape
        if n_codes > n_samples:
            warnings.warn("Code book size reduced to %d" % n_samples)
            n_codes = n_samples
        # uniformly sample the code vectors
        c_ind = np.arange(0, n_codes) * n_samples / n_codes # round to integer
        self.code_book = x[c_ind, :]

    def train_Lloyd(self, x):
        """
        Train Vector Quantizer for given training data for min Euclidean square sum distortion.
        Input:
        ------
        x: training data. [n_samples, n_features]
        Return:
        ------
        var_c: [n_codes, n_features]. Square deviations for each VQ cluster center.
        The size of the code book might be reduced after training.

        """
        if x.shape[1] != self.code_book.shape[1]:
            raise ValueError("Incompatible vector length")
        init_codebook_size = self.code_book.shape[0]
        n_samples, n_features = x.shape
        if n_samples < init_codebook_size * 2:
            warnings.warn("Too few training data. Results may be inaccurate")
        # initial encoding
        ix = self.encode(x)
        ix_previous = np.zeros((n_samples))
        while np.any(ix != ix_previous):
            for n in range(0, self.code_book.shape[0]):
                # modify codebook entry
                self.code_book[n, :] = np.mean(x[ix == n, :], axis=0) # might be empty, mean of empty is nan
            # remove nan codeword, the codebook size might be reduced.
            self.code_book = self.code_book[np.isfinite(self.code_book[:, 0]), :]
            ix_previous = ix
            ix = self.encode(x) # encode with new codebook
        # Should I warn if codebook has reduced?
        # Compute mean square deviations from cluster center.
        var_c = np.zeros(self.code_book.shape)
        for n in range(0, self.code_book.shape[0]):
            devs = x[ix == n, :] - np.tile(self.code_book[n, :], (np.sum(ix == n), 1))
            var_c[n, :] = np.mean(np.multiply(devs, devs), axis=0) #[n_features, ]
        return var_c

    def encode(self, x):
        """
        Encode continuous valued vectors by integer codes.
        Input:
        ------
        x: continuous vectors. [n_samples, n_features]
        Return:
        ------
        ind_x: [n_samples, ]. Indices of the nearest codebook point for each vector.
               Index start from 0.
        """
        n_samples, n_features = x.shape
        C = self.code_book # [n_codes, n_features]
        # for each xk in x, find nearest codeword c in C: 
        # distance(x, c): (xk - c) * (xk - c)^T = xk * xk' - 2 * xk * c' + c * c'
        c_inner_prod = np.sum(np.multiply(C, C), axis=1) # [n_codes, ]
        xc_inner_prod = x.dot(C.T) # [n_samples, n_codes], row k is [xk * ci^T for ci in all codewords]
        dist = np.tile(c_inner_prod, (n_samples, 1)) - 2 * xc_inner_prod # [n_samples, n_codes]
        ind_x = np.argmin(dist, axis=1)
        return ind_x

    def decode(self, ind_x):
        """
        Recreate continous-valued vectors from the discrete integer codes.
        Input:
        ------
        ind_x: [n_samples, ] code indices.
        Return:
        ------
        x: [n_samples, n_features]. Continous-valued vectors. 
        """
        if np.any(ind_x < 0) or np.any(ind_x >= len(self.code_book.shape[0])):
            raise ValueError("index value must be within codebook range")
        else:
            return self.code_book[ind_x, :]

def make_VQ(x, n_codes):
    """
    Create codebook using n_codes codewords based on training data x, using init_by_data and train_Lloyd.
    Input:
    ------
    x: continuous vectors. [n_samples, n_features]
    n_codes: desired size of code book, must satisfy n_codes <= n_samples
    Return:
    ------
    vq: the VectorQuantizer object
    [n_codes_aftertrain, n_features]. Square deviations for each VQ cluster center.
    """
    vq = VectorQuantizer()
    vq.init_by_data(x, n_codes)
    var_c = vq.train_Lloyd(x)
    return vq, var_c

def gather_codebook(vq):
    """
    Gather codebooks in a single or list of VQ objects.
    Input: 
    -------
    vq: a single or a list of VectorQuantizer object(s).
    Return:
    codebooks: list of length n_obj, each element is a codebook 2D array.
    -------
    """
    if isinstance(vq, VectorQuantizer):
        vq = [vq]
    if isinstance(vq, list) and isinstance(vq[0], VectorQuantizer):
        codebooks = []
        for i in range(0, len(vq)):
            codebooks.append(vq[i].code_book[:])
        return codebooks
    else:
        raise ValueError("Input must be a VQ object or a list of VQ objects")
