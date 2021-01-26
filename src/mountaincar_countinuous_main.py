import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow_probability.python.distributions import Normal


class HiddenLayer:
    """
    This class implements a fully connected neural network layer.
    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation_func=tf.nn.tanh,
                 use_bias: bool = True):
        self.W = tf.Variable(tf.random_normal_initializer(shape=[input_size, output_size]))


if __name__ == '__main__':
    var1 = tf.Variable()
    print(var1)