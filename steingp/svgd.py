import tensorflow as tf
import tensorflow_probability as tfp
from steingp.kernel import K_and_dKx, Kernel
from time import time
from tensorflow import Tensor
from typing import Union
from steingp.gp import SteinGPR, SteinMC, update_gpmc, update_gpr, basic_score, SteinSGPR
from tqdm import trange


class SVGD:
    def __init__(self,
                 model: Union[SteinGPR, SteinMC],
                 kernel: Kernel,
                 n_particles: int,
                 theta: Tensor = None,
                 fudge_factor: float = 1e-6,
                 alpha: float = 0.9,
                 verbose: bool = False,
                 stopping_epsilon: float = 0.1,
                 patience: int = 20):
        self.m = model
        self.kern = kernel
        self.n_particles = n_particles
        self.theta = [self._initialise_theta() if theta is None else theta
                      ][0]  # n_params x n_particles
        self.n_params = self.theta.shape[0]
        self.training_progress = []
        self.historical_grad = tf.constant(0.0, dtype=tf.float64)
        # AdaGrad params
        self.fudge_factor = fudge_factor
        self.alpha = alpha
        self.training_time = None
        self.niter = None
        self.ranger = [trange if verbose else range][0]
        self.stopping_epsilon = stopping_epsilon
        self.patience = patience
        self.grads = None
        self.past_grads = None
        self.bandwidths = []

    def close_model(self):
        expected_theta = tf.reduce_mean(self.theta, axis=1)
        self.m.update(expected_theta)
        return self.m

    def _initialise_theta(self):
        x0 = []
        for p in self.m.trainable_parameters:
            if p.shape == ():
                shp = (1, self.n_particles)
            else:
                shp = (p.shape[0], self.n_particles)

            if p.prior is not None:
                smp = p.prior.sample(shp)
            else:
                smp = tf.random.uniform(shape=shp, dtype=tf.float64)
            x0.append(smp)
        return tf.concat(x0, axis=0)

    def _phi_hat(self, K: Tensor, score_func: Tensor, dK: Tensor):
        return (tf.matmul(K, score_func) + dK) / self.n_particles

    def run(self, iterations: int, stepsize: float = 0.01):
        start = time()
        for i in self.ranger(iterations):
            scores = []
            objs = []
            K, dK = self.kern.gradient(
                self.theta
            )  # TODO: Check output matrices are of correct dimension. Unit test.
            self.bandwidths.append(self.kern.current_h)
            for j in range(self.n_particles):
                self.m.update(self.theta[:, j])
                score, obj = self.m.score()
                scores.append(score)
                objs.append(obj)
            score_fn = tf.stack(scores, axis=0)  # TODO: unit test for this
            grads = self._phi_hat(K, score_fn, dK)
            adj_grad = self.adagrad(grads, i)
            self.theta += stepsize * tf.transpose(adj_grad)
            self.training_progress.append(tf.reduce_mean(objs).numpy())
            if i > 50:
                diff = []
                for i in range(1, self.patience):
                    a = self.training_progress[-i]
                    b = self.training_progress[-i - 1]
                    diff.append(tf.abs(a - b))
                incremants = tf.reduce_mean(diff)
                if incremants < self.stopping_epsilon:
                    break

        self.training_time = time() - start
        self.niter = iterations

    def adagrad(self, grads, nit: int):
        """
        Use AdaGrad to compute the learning rate.
        """
        grad_square = tf.math.square(grads)
        if nit == 0:
            self.historical_grad += grad_square
        else:
            self.historical_grad = (
                tf.multiply(self.historical_grad, self.alpha) +
                (1 - self.alpha) * grad_square)
        adj_grad = tf.divide(grads,
                             self.fudge_factor + tf.sqrt(self.historical_grad))
        return adj_grad


class SparseSVGD:
    def __init__(self,
                 model: SteinSGPR,
                 kernel: Kernel,
                 n_particles: int,
                 fudge_factor: float = 1e-6,
                 alpha: float = 0.9,
                 verbose: bool = False,
                 stopping_epsilon: float = 0.1,
                 patience: int = 20):
        self.m = model
        self.kern = kernel
        self.n_particles = n_particles
        self.theta= self._initialise_theta()
        self.n_params = self.theta[0].shape[0] + self.theta[1].shape[0]
        self.training_progress = []
        self.historical_grad_h = tf.constant(0.0, dtype=tf.float64)
        # self.Z_n = model.inducing_variable.Z.shape[0]
        # self.Z_d = model.inducing_variable.Z.shape[1]
        # AdaGrad params
        self.fudge_factor = fudge_factor
        self.alpha = alpha
        self.training_time = None
        self.niter = None
        self.ranger = [trange if verbose else range][0]
        self.stopping_epsilon = stopping_epsilon
        self.patience = patience
        self.grads = None
        self.past_grads = None

    def close_model(self):
        h = tf.reduce_mean(self.theta, axis=1)
        self.m.update_h(h)
        return self.m

    def _initialise_theta(self):
        htheta = self._initialise_h_theta()
        return htheta

    def _initialise_h_theta(self):
        x0 = []
        for p in self.m.trainable_parameters:
            if p.shape == ():
                shp = (1, self.n_particles)
            else:
                shp = (p.shape[0], self.n_particles)

            if p.prior is not None:
                smp = p.prior.sample(shp)
            else:
                smp = tf.random.uniform(shape=shp, dtype=tf.float64)
            x0.append(smp)
        return tf.concat(x0, axis=0)

    def _phi_hat(self, K: Tensor, score_func: Tensor, dK: Tensor):
        return (tf.matmul(K, score_func) + dK) / self.n_particles

    def update_hyper(self, stepsize: float, nit: int):
        scores = []
        objs = []
        K, dK = self.kern.gradient(
            self.theta)  # TODO: Work out how to compute K
        assert K.shape == (self.n_particles, self.n_particles)
        for j in range(self.n_particles):
            self.m.update_h(self.theta[:, j])  # TODO: Only update Z
            score, obj = self.m.score()
            scores.append(score)
            objs.append(obj)
        score_fn = tf.stack(scores, axis=0)  # TODO: unit test for this
        grads = self._phi_hat(K, score_fn, dK)
        self.training_progress.append(tf.reduce_mean(objs).numpy())
        adj_grad = self.adagrad(grads, nit, self.historical_grad_h)
        self.theta += stepsize * tf.transpose(adj_grad)

    def run(self, iterations: int, stepsize: float = 0.05):
        start = time()
        for i in self.ranger(iterations):
            self.update_hyper(stepsize, i)
            if i > 50:
                diff = []
                for i in range(1, self.patience):
                    a = self.training_progress[-i]
                    b = self.training_progress[-i - 1]
                    diff.append(tf.abs(a - b))
                incremants = tf.reduce_mean(diff)
                if incremants < self.stopping_epsilon:
                    break
        self.training_time = time() - start
        self.niter = iterations

    def adagrad(self, grads, nit: int, hist_grad):
        """
        Use AdaGrad to compute the learning rate.
        """
        grad_square = tf.math.square(grads)
        if nit == 0:
            hist_grad += grad_square
        else:
            hist_grad = (tf.multiply(hist_grad, self.alpha) +
                         (1 - self.alpha) * grad_square)
        adj_grad = tf.divide(grads, self.fudge_factor + tf.sqrt(hist_grad))
        return adj_grad


def run_svgd(model, iterations, theta, partitions, stepsize=0.05):
    n_particles = theta.shape[1]
    n_params = theta.shape[0]  # TODO: Have this extracted from model.
    logf = []
    historical_grad = tf.constant(0.0, dtype=tf.float64)
    for i in range(iterations):
        scores = []
        objs = []
        Kxy, dKxy = K_and_dKx(theta)
        # if i % 10 == 0:
        #     plot_K(Kxy, dKxy, i, "gpmc_exponential/{}.png".format(int(i/10)))
        assert Kxy.shape == (n_particles, n_particles)
        assert dKxy.shape == (n_particles, n_params)
        for j in range(n_particles):
            model = update_gpmc(model, theta[:, j], partitions)
            score, obj = basic_score(model)  #, theta[:, j])
            # if model.V.trainable:
            #     score = tf.squeeze(
            #         tf.concat((score[0], tf.expand_dims(score[1:], axis=1)),
            #                   axis=0))
            # else:
            #     score = tf.stack(score, axis=0)
            scores.append(score)
            objs.append(obj)
        score_fn = tf.stack(scores, axis=0)
        assert score_fn.shape == (n_particles, n_params)
        grads = (tf.matmul(Kxy, score_fn) + dKxy) / n_particles
        adj_grad, historical_grad = adagrad(historical_grad, grads, i)
        theta += stepsize * tf.transpose(adj_grad)
        logf.append(tf.reduce_mean(objs).numpy())
    return theta, logf


def adagrad(historical_grad,
            grads,
            nit: int,
            fudge_factor: float = 1e-6,
            alpha: float = 0.9):
    """
    Use AdaGrad to compute the learning rate.
    """
    grad_square = tf.math.square(grads)
    if nit == 0:
        historical_grad += grad_square
    else:
        historical_grad = (tf.multiply(historical_grad, alpha) +
                           (1 - alpha) * grad_square)
    adj_grad = tf.divide(grads, fudge_factor + tf.sqrt(historical_grad))
    return adj_grad, historical_grad


def initialise_particles(model, n_particles: int = 1):
    """
    Initialise a set of particles based upon the number of parameters in a model, and the desired number of particles.
    """
    x0 = []
    for p in model.trainable_parameters:
        if p.shape == ():
            shp = (1, n_particles)
        else:
            shp = (p.shape[0], n_particles)

        if p.prior is not None:
            smp = p.prior.sample(shp)
        else:
            smp = tf.random.uniform(shape=shp, dtype=tf.float64)
        x0.append(smp)
    return tf.concat(x0, axis=0)
