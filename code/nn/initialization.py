import random
import numpy as np
import tensorflow as tf


default_rng = np.random.RandomState(random.randint(0, 9999))


def random_init(size, rng=None, rng_type=None):
    """
        Return initial parameter values of the specified size

        Inputs
        ------

            size            : size of the parameter, e.g. (100, 200) and (100,)
            rng             : random generator; the default is used if None
            rng_type        : the way to initialize the values
                                None    -- (default) uniform [-0.05, 0.05]
                                normal  -- Normal distribution with unit variance and zero mean
                                uniform -- uniform distribution with unit variance and zero mean
    """
    if rng is None: rng = default_rng
    if rng_type is None:
        vals = rng.uniform(low=-0.05, high=0.05, size=size)
    elif rng_type == "normal":
        vals = rng.standard_normal(size)
    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)
    else:
        raise Exception(
            "unknown random inittype: {}".format(rng_type)
          )
    return vals.astype(np.float32)


# def create_shared(vals, name=None):
#     """
#         return a tf variable with initial values as vals
#     """
#     return tf.Variable(vals, name=name)


def linear(x):
    return x


def get_activation_by_name(name):
    if name.lower() == "relu":
        return tf.nn.relu  # usage tf.nn.relu(features, name=None)
    elif name.lower() == "sigmoid":
        return tf.sigmoid  # usage tf.sigmoid(x, name=None)
    elif name.lower() == "tanh":
        return tf.tanh  # usage tf.tanh(x, name=None)
    elif name.lower() == "softmax":
        return tf.nn.softmax  # usage softmax(logits, dim=-1, name=None)
    # elif name.lower() == "none" or name.lower() == "linear":
    #     return linear
    else:
        raise Exception("unknown activation type: {}".format(name))


def init_w_b_vals(w_shape, b_shape, activation):
    w_vals = tf.random_uniform(w_shape, minval=-0.5, maxval=0.5)
    if activation == 'softmax':
        w_vals *= 0.001
    if activation == 'relu':
        b_vals = np.ones(b_shape) * 0.01
    else:
        b_vals = tf.random_uniform(shape=b_shape, minval=-0.5, maxval=0.5)
    return w_vals, b_vals
