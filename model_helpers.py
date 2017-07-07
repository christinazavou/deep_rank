
import tensorflow as tf
import numpy as np


def cosine_distance(vec1, vec2):
	# dotprod = tf.matmul(tf.reshape(vec1, [1, -1]), tf.reshape(vec2, [-1, 1]))
	# vec1_magn = tf.sqrt(tf.reduce_sum(
	# 	tf.matmul(tf.reshape(vec1, [1, -1]), tf.reshape(vec1, [-1, 1]))))
	# vec2_magn = tf.sqrt(tf.reduce_sum(
	# 	tf.matmul(tf.reshape(vec2, [1, -1]), tf.reshape(vec2, [-1, 1]))))
	# result = tf.div(dotprod, tf.multiply(vec1_magn, vec2_magn))
	# return result
	distance = tf.reduce_sum(tf.multiply(vec1, vec2), 1, keep_dims=True)
	distance = tf.div(
		distance,
		tf.multiply(
			tf.sqrt(tf.reduce_sum(tf.square(vec1), 1, keep_dims=True)),
			tf.sqrt(tf.reduce_sum(tf.square(vec2), 1, keep_dims=True))
		)
	)
	distance = tf.reshape(distance, [-1], name="distance")
	return distance


# def batch_max_margin_loss(sim1, sim2, y, margin=tf.constant(0.009)):
# 	# sim1 is the similarity of Q+ to Q
# 	# sim2 is the similarity of Q? to Q
# 	# y contains 0 if Q? is equal with Q+, 1 otherwise
# 	y = y * margin
# 	loss = sim1 - sim2 + y
# 	loss = tf.reduce_sum(loss)
# 	return loss
def batch_max_margin_loss(sim1, sim2, margin=tf.constant(0.009)):
	# sim1 is the similarity of Q+ to Q
	# sim2 is the similarity of Q? to Q
	# y contains 0 if Q? is equal with Q+, 1 otherwise
	loss = margin - sim1 + sim2
	loss = tf.maximum(tf.zeros_like(sim1), loss)
	loss = tf.reduce_sum(loss)
	return loss


if __name__ == '__main__':

	with tf.Session() as sess:

		# x1 = tf.placeholder(tf.float32, [None, 4])
		# x2 = tf.placeholder(tf.float32, [None, 4])
		# print sess.run(
		# 	[cosine_distance(x1, x2)],
		# 	feed_dict={
		# 		x1: np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
		# 		x2: np.array([[4, 3, 1, 1], [1, 2, 3, 4]])
		# 	})

		# y_ = tf.placeholder(tf.float32, [None, 1])
		x1 = tf.placeholder(tf.float32, [None, 1])
		x2 = tf.placeholder(tf.float32, [None, 1])
		print sess.run(
			# [batch_max_margin_loss(x1, x2, y_)],
			[batch_max_margin_loss(x1, x2)],
			feed_dict={
				x1: np.array([[0.8], [0.9], [0.99]]),
				x2: np.array([[0.5], [0.7], [0.99]])
				# y_: np.array([[1], [1], [0]])
			}
		)
