import numpy as np
import tensorflow as tf

from baselines.common.distributions import make_pdtype

def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(
        inputs=inputs,
        kernel_size=kernel_size,
        strides=(strides, strides),
        activation=tf.nn.relu,
        kernel_initializer=tf.orthogonal_initializer(gain=gain)
    )