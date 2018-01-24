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


def devloss0(labels, scores):  # OK
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


def dev_entropy_loss(target_scores, output_scores):
    # max(x, 0) - x * z + log(1 + exp(-abs(x))) for stability and overflow avoidance
    target_scores = np.array(target_scores, np.float32)
    output_scores = np.array(output_scores, np.float32)
    x_entropy = np.max(output_scores, 0) - output_scores + target_scores + np.log(1+np.exp(-abs(output_scores)))
    return np.mean(x_entropy)

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
