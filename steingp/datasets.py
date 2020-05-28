import numpy as np
from numpy.random import RandomState
from numpy import ndarray
from gpflow.kernels import SquaredExponential
import tensorflow as tf
import tensorflow_probability as tfp
from steingp.kernel import stabilise
from steingp.utils import train_test_idx


def exponential_lik(X: ndarray, rng: RandomState):
    return rng.exponential(X)


def exponential_gen(n: int):
    rng = np.random.RandomState(123)
    Xfull = np.sort(rng.uniform(low=-3, high=3, size=(n * 10, 1)), axis=0)
    Yfull = exponential_lik(np.sin(Xfull)**2, rng)
    train_idx, _ = train_test_idx(n, rng)
    X = Xfull[train_idx, :]
    Y = Yfull[train_idx, :]
    return X, Y, Xfull, Yfull


def bernoulli_gen(n: int, train_prop: float = 0.7):
    rng = np.random.RandomState(123)
    Xfull = np.sort(np.sin(rng.uniform(low=0, high=10, size=(n, 1)))**2,
                    axis=0)
    noise = rng.normal(loc=0, scale=0.05, size=Xfull.shape)
    Yfull = np.where(np.logical_and(0.3 < Xfull + noise, Xfull + noise < 0.7),
                     0, 1)
    train_idx, test_idx = train_test_idx(n, rng, train_prop)
    X = Xfull[train_idx, :]
    Y = Yfull[train_idx, :]
    print("Proportion of positive labels in training data: {}".format(
        np.mean(Y)))
    Xte = Xfull[test_idx, :]
    Yte = Yfull[test_idx, :]
    return X, Y, Xte, Yte


def gen_gp_data(X, lengthscale=0.5, variance=0.2, obs_noise=0.1):
    seed = tfp.util.SeedStream(123, salt="MVN")
    n = X.shape[0]
    kern = SquaredExponential(lengthscales=lengthscale, variance=variance)
    K = stabilise(kern.K(X, X))
    generator = tfp.distributions.MultivariateNormalFullCovariance(
        tf.reshape(tf.zeros_like(X), [-1]), K)
    y = tf.reshape(generator.sample(seed=seed()), (n, 1)).numpy()
    return tf.reshape(
        y + tf.cast(
            tfp.distributions.Normal(0, obs_noise).sample(
                y.shape, seed=seed()), tf.float64), (n, 1)).numpy()
