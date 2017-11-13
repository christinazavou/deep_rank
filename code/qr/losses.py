import tensorflow as tf
import numpy as np


LEN_NEGATIVES = 20


# sess = tf.Session()
#
# qpp = tf.constant([[0, 1, 2, -1, -1, -1, -1, -1, -1, -1], [0, 1, 2, -1, -1, -1, -1, -1, -1, -1]])
# # [num_of_tuples]
# pos_scores = tf.constant([0.9, 0.8, 0.5, 0.9, 0.8, 0.5])
# # [num_of_tuples, candidate size - 1]
# all_neg_scores = tf.constant([[0.7, 0.6, 0.4, 0.3, 0.2], [0.7, 0.6, 0.4, 0.3, 0.2], [0.7, 0.6, 0.4, 0.3, 0.2],
#                               [0.7, 0.6, 0.4, 0.3, 0.2], [0.7, 0.6, 0.4, 0.3, 0.2], [0.7, 0.6, 0.4, 0.3, 0.2]])
# # [num_of_tuples]
# neg_scores = tf.reduce_max(all_neg_scores, axis=1)
#

def loss0(pos_scores, all_neg_scores):
    # [num_of_tuples]
    neg_scores = tf.reduce_max(all_neg_scores, axis=1)
    diff = neg_scores - pos_scores + 1.0
    # tf.cast((diff > 0), tf.float32) * diff is replacing in matrix diff the values <= 0 with zero
    loss = tf.cast((diff > 0), tf.float32) * diff
    loss = tf.reduce_mean(loss, name='hinge_loss')
    return loss


def loss0sum(pos_scores, all_neg_scores, query_per_pair):
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


def loss1(pos_scores, all_neg_scores, query_per_pair):
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


def loss2(pos_scores, all_neg_scores):
    # [num_of_tuples, candidate size - 1]
    diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.0
    diff = tf.nn.relu(diff)
    loss = tf.reduce_mean(diff, name='hinge_loss')
    return loss


def loss2sum(pos_scores, all_neg_scores, query_per_pair):
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


def devloss0(labels, scores):  # OK
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
        tuples_diff.append(diff.reshape([-1, 1]))
    tuples_diff = np.vstack(tuples_diff)
    tuples_diff = (tuples_diff > 0).astype(np.float32)*tuples_diff
    return np.mean(tuples_diff)


def devloss1(labels, scores):  # OK
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


def devloss2(labels, scores):  # OK
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


def devloss0sum(labels, scores):  # OK
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


def devloss2sum(labels, scores):  # OK
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


# print sess.run(loss0())
# print sess.run(loss1(qpp))
# print sess.run(loss2())
# print sess.run(loss0sum(qpp))
# print sess.run(loss2sum(qpp))


# labels = [np.array([1,1,0,0,1,0,0,0]), np.array([1,1,0,0,1,0,0,0])]
# scores = [np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]), np.array([0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2])]
#
# print devloss0(labels, scores)
# print devloss1(labels, scores)
# print devloss2(labels, scores)
# print devloss0sum(labels, scores)
# print devloss2sum(labels, scores)