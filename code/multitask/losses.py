import tensorflow as tf
import numpy as np


LEN_NEGATIVES = 20


def qrloss0(pos_scores, all_neg_scores):
    # [num_of_tuples]
    neg_scores = tf.reduce_max(all_neg_scores, axis=1)
    diff = neg_scores - pos_scores + 1.0
    # tf.cast((diff > 0), tf.float32) * diff is replacing in matrix diff the values <= 0 with zero
    loss = tf.cast((diff > 0), tf.float32) * diff
    loss = tf.reduce_mean(loss, name='hinge_loss')
    return loss


def qrloss0sum(pos_scores, all_neg_scores, query_per_pair):
    # [num_of_tuples, 20]
    diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.
    diff = tf.nn.relu(diff)
    # [num_of_tuples+1, 20]
    diff = tf.concat([tf.zeros((1, LEN_NEGATIVES)), diff], 0)
    newqpp = query_per_pair + 1
    # [batch_size, 10, 20]
    emb_loss = tf.nn.embedding_lookup(diff, newqpp)
    # [batch_size, 10]
    emb_loss_max = tf.reduce_max(emb_loss, 2)
    # [batch_size]
    loss_pq = tf.reduce_sum(emb_loss_max, 1)
    loss = tf.reduce_mean(loss_pq, name='hinge_loss')
    return loss


def qrloss1(pos_scores, all_neg_scores, query_per_pair):
    # [num_of_tuples, 20]
    diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.
    diff = tf.nn.relu(diff)
    # [num_of_tuples+1, 20]
    diff = tf.concat([tf.zeros((1, LEN_NEGATIVES)), diff], 0)
    newqpp = query_per_pair + 1
    # [batch_size, 10, 20]
    emb_loss = tf.nn.embedding_lookup(diff, newqpp)
    # [batch_size, 10]
    emb_loss_max = tf.reduce_max(emb_loss, 2)
    # [batch_size]
    loss_pq = tf.reduce_max(emb_loss_max, 1)
    loss = tf.reduce_mean(loss_pq, name='hinge_loss')
    return loss


def qrloss2(pos_scores, all_neg_scores):
    # [num_of_tuples, candidate size - 1]
    diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.0
    diff = tf.nn.relu(diff)
    loss = tf.reduce_mean(diff, name='hinge_loss')
    return loss


def qrloss2sum(pos_scores, all_neg_scores, query_per_pair):
    # [num_of_tuples, 20]
    diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.
    diff = tf.nn.relu(diff)
    # [num_of_tuples+1, 20]
    diff = tf.concat([tf.zeros((1, LEN_NEGATIVES)), diff], 0)
    newqpp = query_per_pair + 1
    # [batch_size, 10, 20]
    emb_loss = tf.nn.embedding_lookup(diff, newqpp)
    # [batch_size, 10]
    emb_loss_max = tf.reduce_sum(emb_loss, 2)
    # [batch_size]
    loss_pq = tf.reduce_sum(emb_loss_max, 1)
    loss = tf.reduce_mean(loss_pq, name='hinge_loss')
    return loss


def qrdevloss0(labels, scores):  # OK
    per_query_loss = []
    for query_labels, query_scores in zip(labels, scores):
        pos_scores = [score for label, score in zip(query_labels, query_scores) if label == 1]
        neg_scores = [score for label, score in zip(query_labels, query_scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pos_scores = np.array(pos_scores)
        neg_scores = np.repeat(np.array(neg_scores).reshape([1, -1]), pos_scores.shape[0], 0)
        neg_scores = np.max(neg_scores, 1)
        diff = neg_scores - pos_scores + 1.
        diff = (diff > 0)*diff
        # print 'diff ', diff
        per_query_loss.append(np.mean(diff))
        # print 'ql: ', np.mean(diff)
    # print 'pql: ', np.mean(np.array(per_query_loss))
    return np.mean(np.array(per_query_loss))


def qrdevloss1(labels, scores):  # OK
    query_losses = []
    for query_labels, query_scores in zip(labels, scores):
        pos_scores = [score for label, score in zip(query_labels, query_scores) if label == 1]
        neg_scores = [score for label, score in zip(query_labels, query_scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pos_scores = np.array(pos_scores)
        neg_scores = np.repeat(np.array(neg_scores).reshape([1, -1]), pos_scores.shape[0], 0)
        neg_scores = np.max(neg_scores, 1)
        diff = neg_scores - pos_scores + 1.
        diff = (diff > 0).astype(np.float32)*diff
        query_losses.append(np.max(diff))
    return np.mean(np.array(query_losses))


def qrdevloss2(labels, scores):  # OK
    query_losses = []
    for query_labels, query_scores in zip(labels, scores):
        pos_scores = [score for label, score in zip(query_labels, query_scores) if label == 1]
        neg_scores = [score for label, score in zip(query_labels, query_scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pos_scores = np.array(pos_scores).reshape([-1, 1])
        neg_scores = np.repeat(np.array(neg_scores).reshape([1, -1]), pos_scores.shape[0], 0)
        diff = neg_scores - pos_scores + 1.
        diff = (diff > 0).astype(np.float32)*diff
        query_losses.append(np.mean(diff))
    return np.mean(np.array(query_losses))


def qrdevloss0sum(labels, scores):  # OK
    tuples_diff = []
    for query_labels, query_scores in zip(labels, scores):
        pos_scores = [score for label, score in zip(query_labels, query_scores) if label == 1]
        neg_scores = [score for label, score in zip(query_labels, query_scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pos_scores = np.array(pos_scores)
        neg_scores = np.repeat(np.array(neg_scores).reshape([1, -1]), pos_scores.shape[0], 0)
        neg_scores = np.max(neg_scores, 1)
        diff = neg_scores - pos_scores + 1.
        diff = (diff > 0).astype(np.float32)*diff
        tuples_diff.append(np.sum(diff))
    tuples_diff = np.array(tuples_diff)
    return np.mean(tuples_diff)


def qrdevloss2sum(labels, scores):  # OK
    query_losses = []
    for query_labels, query_scores in zip(labels, scores):
        pos_scores = [score for label, score in zip(query_labels, query_scores) if label == 1]
        neg_scores = [score for label, score in zip(query_labels, query_scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pos_scores = np.array(pos_scores).reshape([-1, 1])
        neg_scores = np.repeat(np.array(neg_scores).reshape([1, -1]), pos_scores.shape[0], 0)
        diff = neg_scores - pos_scores + 1.
        diff = (diff > 0).astype(np.float32)*diff
        query_losses.append(np.sum(diff))
    return np.mean(np.array(query_losses))


def qrdev_entropy_loss(target_scores, output_scores):
    target_scores = np.array(target_scores)
    output_scores = np.array(output_scores)
    print 'len ', target_scores.shape, output_scores.shape
    x_entropy = target_scores * (-np.log(output_scores)) + (1.0 - target_scores) * (-np.log(1.0 - output_scores))
    return np.mean(x_entropy)


def tpentropy_loss(args, target, act_output):
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


# i have 900 tags and only 5 can be positive so taking all (p, n) pairs and averaging is not useful


# if take_max = True, then is equal to loss0, if False, then is equal to loss2
def tphinge_loss(target, act_output, tuples, take_max=False):  # act_output which lies in [0,1]
    act_output = tf.reshape(act_output, [-1, 1])
    tuples_output = tf.nn.embedding_lookup(act_output, tuples)
    tuples_output = tf.squeeze(tuples_output, 2)
    positive_scores = tf.reshape(tuples_output[:, 0], [-1, 1])
    negative_scores = tuples_output[:, 1:]
    diff = negative_scores - positive_scores + 1.
    diff = tf.nn.relu(diff)
    if take_max:
        diff = tf.reduce_max(diff, 1)
    loss = tf.reduce_mean(diff)
    return loss


# if take_max = True, then is equal to loss0, if False, then is equal to loss2
def tpdev_hinge_loss(target, act_output, tuples, take_max=False):  # act_output which lies in [-1,1]
    num_tuples = tuples.shape[0]
    act_output = np.reshape(act_output, [-1, 1])
    tuples = np.reshape(tuples, [-1])
    tuples_output = np.reshape(act_output[tuples], [num_tuples, -1])
    positive_scores = tuples_output[:, 0].reshape([-1, 1])
    negative_scores = tuples_output[:, 1:]
    if take_max:
        negative_scores = np.max(negative_scores).reshape([-1, 1])
    diff = negative_scores - positive_scores + 1.
    diff = (diff > 0)*diff
    loss = np.mean(diff)
    return loss


def tpdev_entropy_loss(args, targets, outputs):
    # outputs are passed through sigmoid, thus they lie in (0,1)
    x_entropy = targets * (-np.log(outputs)) + (1.0 - targets) * (-np.log(1.0 - outputs))
    if 'loss' in args and args.loss == "sum":
        loss = np.sum(np.sum(x_entropy, 1))
    elif 'loss' in args and args.loss == "max":
        loss = np.max(np.sum(x_entropy, 1))
    else:
        loss = np.mean(np.sum(x_entropy, 1))
    return loss
