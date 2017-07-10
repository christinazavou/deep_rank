import tensorflow as tf
from askubuntu.model_helpers import cosine_distance, batch_max_margin_loss


class TextCNN(object):

    def __init__(
			self, sequence_length, num_classes, vocab_size,
			embedding_size, filter_sizes, num_filters, margin=0.009):

    	# Q
		self.input_x = tf.placeholder(
			tf.int32, [None, sequence_length], name="input_x")
		# positive example
		self.input_x1 = tf.placeholder(
			tf.int32, [None, sequence_length], name="input_x1")
		# negative example
		self.input_x2 = tf.placeholder(
			tf.int32, [None, sequence_length], name="input_x2")

		self.dropout_keep_prob = tf.placeholder(
			tf.float32, name="dropout_keep_prob")

		self.margin = tf.constant(margin, tf.float32, name="margin")

		with tf.variable_scope("embedding"):
		    W = tf.get_variable(
		    	name="W",
		        initializer=tf.random_uniform(
		        	[vocab_size, embedding_size], -1.0, 1.0)
		    )
		    with tf.name_scope("embedding_x1"):
			    self.embedded_chars1 = tf.nn.embedding_lookup(W, self.input_x1)
			    self.embedded_chars1_expanded = tf.expand_dims(self.embedded_chars1, -1)
		    tf.get_variable_scope().reuse_variables()
		    with tf.name_scope("embedding_x2"):
			    self.embedded_chars2 = tf.nn.embedding_lookup(W, self.input_x2)
			    self.embedded_chars2_expanded = tf.expand_dims(self.embedded_chars2, -1)
		    # tf.get_variable_scope().reuse_variables()
		    with tf.name_scope("embedding_x"):
			    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
			    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		pooled_outputs, pooled_outputs1, pooled_outputs2 = [], [], []
		for i, filter_size in enumerate(filter_sizes):
			with tf.variable_scope("conv-maxpool-%s" % filter_size):
				# Convolution Layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.get_variable(
					name="W",
					initializer=tf.truncated_normal(filter_shape, stddev=0.1)
				)
				b = tf.get_variable(
					name="b",
					initializer=tf.constant(0.1, shape=[num_filters])
				)
				with tf.name_scope("conv-maxpool-%s-1" % filter_size):
				    conv = tf.nn.conv2d(
				        self.embedded_chars1_expanded,
				        W,
				        strides=[1, 1, 1, 1],
				        padding="VALID",
				        name="conv"
				    )
				    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				    pooled = tf.nn.max_pool(
				        h,
				        ksize=[1, sequence_length - filter_size + 1, 1, 1],
				        # size of the window for each dimension of the input tensor.
				        strides=[1, 1, 1, 1],
				        padding='VALID',
				        name="pool"
				    )
				    pooled_outputs1.append(pooled)
				tf.get_variable_scope().reuse_variables()
				with tf.name_scope("conv-maxpool-%s-2" % filter_size):
				    conv = tf.nn.conv2d(
				        self.embedded_chars2_expanded,
				        W,
				        strides=[1, 1, 1, 1],
				        padding="VALID",
				        name="conv"
				    )
				    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				    pooled = tf.nn.max_pool(
				        h,
				        ksize=[1, sequence_length - filter_size + 1, 1, 1],
				        # size of the window for each dimension of the input tensor.
				        strides=[1, 1, 1, 1],
				        padding='VALID',
				        name="pool"
				    )
				    pooled_outputs2.append(pooled)
				with tf.name_scope("conv-maxpool-%s-" % filter_size):
				    conv = tf.nn.conv2d(
				        self.embedded_chars_expanded,
				        W,
				        strides=[1, 1, 1, 1],
				        padding="VALID",
				        name="conv"
				    )
				    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				    pooled = tf.nn.max_pool(
				        h,
				        ksize=[1, sequence_length - filter_size + 1, 1, 1],
				        # size of the window for each dimension of the input tensor.
				        strides=[1, 1, 1, 1],
				        padding='VALID',
				        name="pool"
				    )
				    pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool1 = tf.concat(pooled_outputs1, 3)
		self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, num_filters_total])
		self.h_pool2 = tf.concat(pooled_outputs2, 3)
		self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, num_filters_total])
		self.h_pool = tf.concat(pooled_outputs, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
		# reshape so that we have shape [batch, num_features_total]

		# Add dropout
		with tf.name_scope("dropout-1"):
		    self.h_drop1 = tf.nn.dropout(self.h_pool1_flat, self.dropout_keep_prob)
		with tf.name_scope("dropout-2"):
		    self.h_drop2 = tf.nn.dropout(self.h_pool2_flat, self.dropout_keep_prob)
		with tf.name_scope("dropout-"):
		    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		with tf.name_scope("output"):
			self.similarity1 = cosine_distance(self.h_drop1, self.h_drop)
			self.similarity2 = cosine_distance(self.h_drop2, self.h_drop)

			similarities = tf.stack(
				(self.similarity1, self.similarity2), 1, name="similarities")

			# self.similarities = similarities  # temp

			self.predictions = tf.argmax(similarities, 1, name="predictions")

		with tf.name_scope("loss"):
		    self.loss = batch_max_margin_loss(
		    	self.similarity1, self.similarity2, self.margin)

		with tf.name_scope("accuracy"):
		    correct_predictions = tf.equal(
		    	self.predictions, tf.zeros_like(self.predictions)
		    )

		    # self.correct_predictions = correct_predictions  # temp

		    self.accuracy = tf.reduce_mean(
		    	tf.cast(correct_predictions, "float"), name="accuracy"
		    )

	# def init_copied():

	# 	with tf.variable_scope("embedding"):
	# 	    W = tf.get_variable(
	# 	    	name="W",
	# 	        initializer=tf.random_uniform(
	# 	        	[vocab_size, embedding_size], -1.0, 1.0)
	# 	    )
	# 	    with tf.name_scope("embedding_x1"):
	# 		    self.embedded_chars1 = tf.nn.embedding_lookup(W, self.input_x1)
	# 		    self.embedded_chars1_expanded = tf.expand_dims(self.embedded_chars1, -1)
	# 	    tf.get_variable_scope().reuse_variables()
	# 	    with tf.name_scope("embedding_x2"):
	# 		    self.embedded_chars2 = tf.nn.embedding_lookup(W, self.input_x2)
	# 		    self.embedded_chars2_expanded = tf.expand_dims(self.embedded_chars2, -1)
	# 	    # tf.get_variable_scope().reuse_variables()
	# 	    with tf.name_scope("embedding_x"):
	# 		    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
	# 		    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

	# 	pooled_outputs, pooled_outputs1, pooled_outputs2 = [], [], []
	# 	for i, filter_size in enumerate(filter_sizes):
	# 		with tf.variable_scope("conv-maxpool-%s" % filter_size):
	# 			# Convolution Layer
	# 			filter_shape = [filter_size, embedding_size, 1, num_filters]
	# 			W = tf.get_variable(
	# 				name="W",
	# 				initializer=tf.truncated_normal(filter_shape, stddev=0.1)
	# 			)
	# 			b = tf.get_variable(
	# 				name="b",
	# 				initializer=tf.constant(0.1, shape=[num_filters])
	# 			)
	# 			with tf.name_scope("conv-maxpool-%s-1" % filter_size):
	# 			    conv = tf.nn.conv2d(
	# 			        self.embedded_chars1_expanded,
	# 			        W,
	# 			        strides=[1, 1, 1, 1],
	# 			        padding="VALID",
	# 			        name="conv"
	# 			    )
	# 			    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
	# 			    pooled = tf.nn.max_pool(
	# 			        h,
	# 			        ksize=[1, sequence_length - filter_size + 1, 1, 1],
	# 			        # size of the window for each dimension of the input tensor.
	# 			        strides=[1, 1, 1, 1],
	# 			        padding='VALID',
	# 			        name="pool"
	# 			    )
	# 			    pooled_outputs1.append(pooled)
	# 			tf.get_variable_scope().reuse_variables()
	# 			with tf.name_scope("conv-maxpool-%s-2" % filter_size):
	# 			    conv = tf.nn.conv2d(
	# 			        self.embedded_chars2_expanded,
	# 			        W,
	# 			        strides=[1, 1, 1, 1],
	# 			        padding="VALID",
	# 			        name="conv"
	# 			    )
	# 			    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
	# 			    pooled = tf.nn.max_pool(
	# 			        h,
	# 			        ksize=[1, sequence_length - filter_size + 1, 1, 1],
	# 			        # size of the window for each dimension of the input tensor.
	# 			        strides=[1, 1, 1, 1],
	# 			        padding='VALID',
	# 			        name="pool"
	# 			    )
	# 			    pooled_outputs2.append(pooled)
	# 			with tf.name_scope("conv-maxpool-%s-" % filter_size):
	# 			    conv = tf.nn.conv2d(
	# 			        self.embedded_chars_expanded,
	# 			        W,
	# 			        strides=[1, 1, 1, 1],
	# 			        padding="VALID",
	# 			        name="conv"
	# 			    )
	# 			    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
	# 			    pooled = tf.nn.max_pool(
	# 			        h,
	# 			        ksize=[1, sequence_length - filter_size + 1, 1, 1],
	# 			        # size of the window for each dimension of the input tensor.
	# 			        strides=[1, 1, 1, 1],
	# 			        padding='VALID',
	# 			        name="pool"
	# 			    )
	# 			    pooled_outputs.append(pooled)

	# 	# Combine all the pooled features
	# 	num_filters_total = num_filters * len(filter_sizes)
	# 	self.h_pool1 = tf.concat(pooled_outputs1, 3)
	# 	self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, num_filters_total])
	# 	self.h_pool2 = tf.concat(pooled_outputs2, 3)
	# 	self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, num_filters_total])
	# 	self.h_pool = tf.concat(pooled_outputs, 3)
	# 	self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
	# 	# reshape so that we have shape [batch, num_features_total]

	# 	# Add dropout
	# 	with tf.name_scope("dropout-1"):
	# 	    self.h_drop1 = tf.nn.dropout(self.h_pool1_flat, self.dropout_keep_prob)
	# 	with tf.name_scope("dropout-2"):
	# 	    self.h_drop2 = tf.nn.dropout(self.h_pool2_flat, self.dropout_keep_prob)
	# 	with tf.name_scope("dropout-"):
	# 	    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

	# 	with tf.name_scope("output"):
	# 		self.similarity1 = cosine_distance(self.h_drop1, self.h_drop)
	# 		self.similarity2 = cosine_distance(self.h_drop2, self.h_drop)

	# 		similarities = tf.stack(
	# 			(self.similarity1, self.similarity2), 1, name="similarities")

	# 		# self.similarities = similarities  # temp

	# 		self.predictions = tf.argmax(similarities, 1, name="predictions")

	# 	with tf.name_scope("loss"):
	# 	    self.loss = batch_max_margin_loss(
	# 	    	self.similarity1, self.similarity2, self.margin)

	# 	with tf.name_scope("accuracy"):
	# 	    correct_predictions = tf.equal(
	# 	    	self.predictions, tf.zeros_like(self.predictions)
	# 	    )

	# 	    # self.correct_predictions = correct_predictions  # temp

	# 	    self.accuracy = tf.reduce_mean(
	# 	    	tf.cast(correct_predictions, "float"), name="accuracy"
	# 	    )
