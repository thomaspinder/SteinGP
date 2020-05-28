import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Union
from tensorflow import Tensor
from numpy import ndarray, pi
from .bandwidths import Bandwidth
from gpflow.kernels import Matern32
from gpflow.utilities.ops import square_distance


class Kernel:
    def __init__(self, name=None):
        self.name = name
        self.current_h = None

    def k(self, theta):
        raise NotImplementedError

    @staticmethod
    def _pariwise_distance(X):
        """
        Compute the pairwise distance between all pairs of a matrix. Uses the result that the D = r-2*AA'+r'
        where r is the squared norm of the input matrix A. Result taken from:
        https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor
        -in-tensorflow
        """
        norm = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1, 1])
        return norm - 2 * tf.matmul(X, X,
                                    transpose_b=True) + tf.transpose(norm)

    @staticmethod
    def _stabilise(x: Union[ndarray, Tensor], noise: float = 1e-6):
        n = x.shape[0]
        x_diag = tf.linalg.diag_part(x)
        noise = tf.fill([n], tf.constant(noise, dtype=tf.float64))
        return tf.linalg.set_diag(x, x_diag + noise)

    def gradient(self, theta):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            K = self.k(theta)
            dK = tf.transpose(tape.gradient(K, theta))
        # dK = tape.gradient(K, theta)
        return K, dK


class RBF(Kernel):
    def __init__(self, h: Bandwidth, name="RBF"):
        self.h = h
        super().__init__(name)

    def k(self, theta: Tensor):
        if theta.ndim == 2:
            pdist = pariwise_distance(tf.transpose(theta))
        else:
            pdist = self.slow_pdist(tf.transpose(theta))

        bw = self.h.estimate(pdist)
        self.current_h = bw
        return tf.exp(-pdist / bw**2 / 2.0)

    def slow_pdist(self, theta: Tensor):
        pdist = []
        for i in range(theta.shape[0]):
            row = []
            for j in range(theta.shape[0]):
                row.append(
                    tf.square(tf.linalg.norm(theta[i, :, :] - theta[j, :, :])))
            pdist.append(tf.convert_to_tensor(row))
        return tf.stack(pdist, axis=1)


class IMQ(Kernel):
    def __init__(self, c: float, beta: float):
        self.c = c
        assert beta < 0, "Please supply a beta value < 0"
        self.beta = beta
        super().__init__(name='IMQ')

    def k(self, theta: Tensor):
        pdist = pariwise_distance(tf.transpose(theta))
        l2_pdist = tf.sqrt(tf.square(pdist))
        c = self.c**2
        p = l2_pdist + c
        K = tf.pow(p, self.beta)
        return K


class MaternWrapper(Kernel):
    def __init__(self, h: Bandwidth, variance: float):
        super().__init__(name="Matern32")
        self.kernel = Matern32(variance=variance)
        self.h = h

    def k(self, theta: Tensor):
        t = tf.transpose(theta)
        pdist = square_distance(t, t)
        bw = self.h.estimate(pdist)
        self.kernel.lengthscales.assign(bw)
        self.current_h = bw
        K = self.kernel.K_r(pdist)
        return K


class RFF:
    def __init__(self, n_features, noise=1.0):
        self.n_features = int(n_features)
        self.sigma = noise
        self.w = None
        self.b = None
        self.kernel_ = None

    def fit(self, X):
        N = X.shape[0]
        Z, w, b = self._compute_features(X, return_vars=True)
        sigma_I = tf.cast(self.sigma * tf.eye(N), tf.float64)
        self.kernel_ = tf.matmul(Z, Z, transpose_a=True) + sigma_I

    def _compute_features(self, X, return_vars):
        N, D = X.shape
        D = int(D)
        if self.w is not None:
            W = self.w
            b = self.b
        else:
            W = tf.cast(
                tfp.distributions.Normal(
                    loc=0.0,
                    scale=1.0).sample(sample_shape=(self.n_features, D)),
                tf.float64)
            b = tf.cast(
                tfp.distributions.Uniform(low=0.0,
                                          high=2 * tf.constant(pi)).sample(
                                              sample_shape=(self.n_features,
                                                            1)), tf.float64)
        B = tf.repeat(b, N, axis=1)
        norm = 1. / tf.math.sqrt(tf.cast(self.n_features, tf.float64))
        Z = norm * tf.cast(tf.math.sqrt(2.0), dtype=tf.float64) * tf.math.cos(
            self.sigma * tf.matmul(W, X, transpose_b=True) + B)
        if return_vars:
            return Z, W, b
        return Z

    def get_kernel(self):
        return self.kernel_


class FourierRBF(Kernel):
    def __init__(self, RFF, name: str = "Fourier"):
        super().__init__(name)
        self.rff = RFF

    def k(self, theta: Tensor):
        self.rff.fit(tf.transpose(theta))
        return self.rff.get_kernel()


def kernel(X):
    """
    Compute the kernel function over a set of particles.
    """
    pdist = pariwise_distance(X)
    bw = median_trick(pdist)
    return tf.exp(-pdist / bw**2 / 2.0)


def K_and_dKx(theta):
    """
    Compute the derivative of the kernel function w.r.t a set of particles theta using TensorFlow's AutoDiff
    functionality.
    """
    with tf.GradientTape() as tape:
        tape.watch(theta)
        Kxy = kernel(tf.transpose(theta))
    dKxy = tf.transpose(tape.gradient(Kxy, theta))
    return Kxy, dKxy


def stabilise(x: Union[ndarray, Tensor], noise: float = 1e-6):
    n = x.shape[0]
    x_diag = tf.linalg.diag_part(x)
    noise = tf.fill([n], tf.constant(noise, dtype=tf.float64))
    return tf.linalg.set_diag(x, x_diag + noise)


def median_trick(D: tf.Tensor):
    """
    Compute the median trick to estimate a kernel bandwidth parameter.
    Args:
        D: Distance matrix

    Returns: Float64
    """
    lower = tfp.stats.percentile(D, 50.0, interpolation="lower")
    upper = tfp.stats.percentile(D, 50.0, interpolation="higher")
    n = D.shape[0]
    median = tf.cast((lower + upper) / 2.0, tf.float64)
    h = tf.sqrt(0.5 * median / tf.cast(tf.math.log(n + 1.0), dtype=tf.float64))
    # Prevent h from being included in the computational graph
    tf.stop_gradient(h)
    return h


def pariwise_distance(X):
    """
    Compute the pairwise distance between all pairs of a matrix. Uses the result that the D = r-2*AA'+r'
    where r is the squared norm of the input matrix A. Result taken from:
    https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor
    -in-tensorflow
    """
    ndim = X.shape.ndims - 1
    # norm is n_particles x 1
    norm = tf.reshape(tf.reduce_sum(tf.square(X), ndim), [-1, 1])
    # matmul term is n_particle x n_particleZ
    return norm - 2 * tf.matmul(X, X, transpose_b=True) + tf.transpose(norm)


def K_and_dKx(theta):
    """
    Compute the derivative of the kernel function w.r.t a set of particles theta using TensorFlow's AutoDiff
    functionality.
    """
    with tf.GradientTape() as tape:
        tape.watch(theta)
        Kxy = kernel(theta)
    dKxy = tape.gradient(Kxy, theta)
    return Kxy, dKxy
