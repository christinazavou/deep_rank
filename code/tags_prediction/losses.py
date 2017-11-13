import tensorflow as tf
import numpy as np


def entropy_loss(args, target, act_output):
    # Entropy measures the "information" or "uncertainty" of a random variable. When you are using base
    #  2, it is measured in bits; and there can be more than one bit of information in a variable. if
    # x-entropy == 1.15 it means that under the compression the model does on the data, we carry about
    # 1.15 bits of information per sample (need 1.5 bits to represent a sample), on average."""

    w = 1. if 'weight' not in args else args.weight
    weighted_entropy = target * (-tf.log(act_output)) * w + (1.0 - target) * (
    -tf.log(1.0 - act_output))
    weighted_entropy *= considered_examples(target)

    if 'loss' in args and args.loss == "sum":
        return tf.reduce_sum(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')
    elif 'loss' in args and args.loss == "max":
        return tf.reduce_max(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')
    else:
        return tf.reduce_mean(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')


def considered_examples(target):
    return tf.expand_dims(tf.cast(tf.not_equal(tf.reduce_sum(target, 1), 0), tf.float32), 1)


def hinge_loss():
    raise Exception()


def dev_entropy_loss(args, targets, outputs):
    # outputs are passed through sigmoid, thus they lie in (0,1)
    x_entropy = targets * (-np.log(outputs)) + (1.0 - targets) * (-np.log(1.0 - outputs))
    if 'loss' in args and args.loss == "sum":
        loss = np.sum(np.sum(x_entropy, 1))
    elif 'loss' in args and args.loss == "max":
        loss = np.max(np.sum(x_entropy, 1))
    else:
        loss = np.mean(np.sum(x_entropy, 1))
    return loss
