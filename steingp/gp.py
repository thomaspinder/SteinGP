import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Tuple, List
from gpflow.models import GPR, GPMC, SGPR, SGPMC
from gpflow.mean_functions import Zero
from gpflow.kernels import Kernel
from gpflow.likelihoods import Likelihood, Gaussian
from typing import Tuple
import numpy as np


class SteinGPR(GPR):
    def __init__(self,
                 data: Tuple,
                 kernel: Kernel,
                 obs_noise: float,
                 partitions: Union[List, Tuple] = None):
        super().__init__(data, kernel, Zero(), obs_noise)
        self.partitions = partitions

    def update(self, theta: Union[Tensor, Tuple]):
        if self.partitions:
            theta = self._slice(theta)
        for p, t in zip(self.trainable_variables, theta):
            t = tf.reshape(t, shape=p.shape)
            p.assign(t)

    def score(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            objective = self.maximum_log_likelihood_objective()
            score = tape.gradient(objective, self.trainable_variables)
        if self.kernel.ard:
            score = tf.squeeze(
                tf.concat((tf.expand_dims(
                    score[0], axis=1), tf.expand_dims(score[1:], axis=1)),
                          axis=0))
        return score, objective

    def _slice(self, theta):
        t_set = []
        counter = 0
        for p in self.partitions:
            t_set.append(theta[counter:(counter + p)])
            counter += p
        return t_set

    def predict(self, Xte: Tensor, theta: np.ndarray, n_samples: int=1):
        samples = []
        for t in theta.T:
            self.update(t)
            pred = self.predict_f_samples(Xte, num_samples=n_samples)
            samples.append(pred.numpy().squeeze())
        return np.vstack(samples)


# TODO: Should probably be SGPMC
class SteinSGPR(SGPMC):
    def __init__(self,
                 data: Tuple,
                 kernel: Kernel,
                 inducing_variables,
                 partitions: Union[List, Tuple],
                 likelihood: Likelihood = Gaussian()):
        super().__init__(data=data,
                         kernel=kernel,
                         likelihood=likelihood,
                         inducing_variable=inducing_variables,
                         mean_function=Zero(),
                         num_latent_gps=1)
        self.partitions = partitions

    def score(self, Z: bool = False):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            objective = self.maximum_log_likelihood_objective()
            score = tape.gradient(objective, self.trainable_variables)
        if self.kernel.ard:
            score = tf.squeeze(tf.concat((score[0], tf.expand_dims(score[1], axis=1), tf.expand_dims(score[2:], axis=1)), axis=0))
        return score, objective

    def _slice(self, theta):
        t_set = []
        counter = 0
        for p in self.partitions:
            t_set.append(theta[counter:(counter + p)])
            counter += p
        return t_set

    def update_h(self, theta: Union[Tensor, Tuple]):
        if self.partitions:
            theta = self._slice(theta)
        for p, t in zip(self.trainable_variables, theta):
            t = tf.reshape(t, shape=p.shape)
            p.assign(t)

    def update_Z(self, theta: Union[Tensor, Tuple]):
        self.trainable_variables[0].assign(theta)

    def predict(self, Xte: Tensor, theta: np.ndarray, n_samples: int=1):
        samples = []
        for t in theta.T:
            self.update_h(t)
            pred = self.predict_f_samples(Xte, num_samples=n_samples)
            samples.append(pred.numpy().squeeze())
        return np.vstack(samples)



class SteinMC(GPMC):
    def __init__(self, data: Tuple, kernel: Kernel, likelihood: Likelihood,
                 partitions: Union[List, Tuple]):
        super().__init__(data, kernel, likelihood, Zero())
        self.partitions = partitions

    def update(self, theta: Tensor):
        theta_set = slice_theta(theta, self.partitions)
        for p, t in zip(self.trainable_variables, theta_set):
            t = tf.reshape(t, shape=p.shape)
            p.assign(t)

    @staticmethod
    def _slice(theta: Tensor, partitions: Union[Tuple, List]):
        t_set = []
        counter = 0
        for p in partitions:
            t_set.append(theta[counter:(counter + p)])
            counter += p
        return t_set

    def score(self):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_variables)
            objective = self.log_posterior_density()
            score_func = tape.gradient(objective, self.trainable_variables)
        if self.V.trainable:
            if self.kernel.ard:
                score = tf.squeeze(
                    tf.concat(
                        (score_func[0], tf.expand_dims(score_func[1], axis=1),
                         tf.expand_dims(score_func[2:], axis=1)),
                        axis=0))
            else:
                score = tf.squeeze(
                    tf.concat(
                        (score_func[0], tf.expand_dims(score_func[1:],
                                                       axis=1)),
                        axis=0))
        else:
            score = tf.stack(score_func, axis=0)
        return score, objective

    def predict(self, Xte: Tensor, theta: np.ndarray, n_samples: int=1):
        samples = []
        for t in theta.T:
            self.update(t)
            pred = self.predict_f_samples(Xte, num_samples=n_samples)
            samples.append(pred.numpy().squeeze())
        return np.vstack(samples)


def basic_score(model: GPMC):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        objective = model.maximum_log_likelihood_objective()
        score_func = tape.gradient(objective, model.trainable_variables)
        if model.V.trainable:
            score = tf.squeeze(
                tf.concat(
                    (score_func[0], tf.expand_dims(score_func[1:], axis=1)),
                    axis=0))
        else:
            score = tf.stack(score_func, axis=0)
    return score, objective


def slice_theta(theta: Tensor, partitions: Union[Tuple, List]):
    t_set = []
    counter = 0
    for p in partitions:
        t_set.append(theta[counter:(counter + p)])
        counter += p
    return t_set


def update_gpmc(m, theta: Tensor, partitions: Union[Tuple, List]):
    theta_set = slice_theta(theta, partitions)
    for p, t in zip(m.trainable_variables, theta_set):
        t = tf.reshape(t, shape=p.shape)
        p.assign(t)
    return m


def update_gpr(model, theta_set):
    for p, t in zip(model.trainable_variables, theta_set):
        p.assign(t)
    return model
