import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import Tensor


class Bandwidth:
    def __init__(self, name=None):
        self.name = name
        self.history = []

    def estimate(self, *args):
        raise NotImplementedError


class Constant(Bandwidth):
    def __init__(self, h: float):
        super().__init__(name='Constant')
        self.h = tf.convert_to_tensor(h, dtype=tf.float64)

    def estimate(self, D: Tensor=None):
        return self.h


class Median(Bandwidth):
    def __init__(self):
        super().__init__(name='Median Trick')

    def estimate(self, D: Tensor):
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
        self.history.append(h)
        return h


class Cyclical(Bandwidth):
    def __init__(self, limits: tuple, starting_iter: int, window: int = 2):
        super().__init__(name='Cyclical')
        self.lower = limits[0]
        self.upper = limits[1]
        self.starting_iter = starting_iter
        self.window_size = window
        self.step = 0
        self.direction = 1
        self.fallback = Median()

    def estimate(self, iteration: int, h: float, D: Tensor):
        if iteration > self.starting_iter:
            if self.direction == 1:
                target = self.upper
            else:
                target = self.lower
            gap = abs(target-h)
            h += self.direction*(self.window_size-self.step)*gap
            if self.step == self.window_size:
                self.step = 0 # Reset progression through window
                self.direction *= -1 # Reverse direction
            else:
                self.step += 1
            return h
        else:
            return self.fallback.estimate(D)
