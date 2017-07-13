# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import time
import numpy as np
import datetime
from deep_rank.deep_ranker import TextCNN
from deep_rank.cnn_config import FLAGS
from deep_rank import load_data_utils


def total_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return total_parameters


this_dir = os.path.dirname(os.path.realpath(__file__))

X, w2i = load_data_utils.train_data_triples(
    title_body=11,
    vocabulary_size=FLAGS.vocabulary_size,
    sequence_length=FLAGS.sequence_length,
    pad=FLAGS.pad
)

trained_dict, trained_embedding_mat = load_data_utils.read_vocabulary()
embedding_mat = load_data_utils.get_embedding_mat(
    trained_dict, trained_embedding_mat, w2i)
vocab_size = len(w2i.keys())
del trained_dict, trained_embedding_mat, w2i

train_batches = load_data_utils.batch_iter(
    X,
    FLAGS.batch_size,
    FLAGS.num_epochs,
    shuffle=True
)

with tf.Graph().as_default():
    savegraph = True
    session_conf = tf.ConfigProto(
    	allow_soft_placement=FLAGS.allow_soft_placement,
    	log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        cnn = TextCNN(
            sequence_length=len(X[0][0]),
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            trained_embeddings=embedding_mat,
            train_embeddings=True
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
        	grads_and_vars, global_step=global_step)

        print 'total params: ', total_parameters()

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.join(this_dir, FLAGS.out_dir, timestamp)
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries

        # dev_loss_batch = tf.placeholder(
        #     tf.float32, [100, 1], name="dev_loss_batch")
        # dev_loss = tf.reduce_mean(dev_loss_batch, name="dev_loss")
        # dev_accuracy_batch = tf.placeholder(
        #     tf.float32, [100, 1], name="dev_accuracy_batch")
        # dev_accuracy = tf.reduce_mean(dev_accuracy_batch, name="dev_accuracy")

        # dev_loss_summary = tf.summary.scalar("dloss", dev_loss)
        # dev_acc_summary = tf.summary.scalar("daccuracy", dev_accuracy)

        # dev_summary_op = tf.summary.merge([dev_loss_summary, dev_acc_summary])
        # dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
        	tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch):
            feed_dict = {
                cnn.input_x: np.asarray([x[0] for x in x_batch]),
            	cnn.input_x1: np.asarray([x[1] for x in x_batch]),
                cnn.input_x2: np.asarray([x[1] for x in x_batch]),
            	cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,
                 cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy: {:g}".format(
            	time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_eval, writer=None):
            data_size = len(x_eval)
            num_batches = 100
            batch_size = 128

            dev_losses = np.zeros((100, 1))
            dev_accuracies = np.zeros((100, 1))

            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                x_batch = x_eval[start_index:end_index]

                feed_dict = {
                	cnn.input_x: np.asarray([x[0] for x in x_batch]),
                    cnn.input_x1: np.asarray([x[1] for x in x_batch]),
                    cnn.input_x2: np.asarray([x[2] for x in x_batch]),
                	cnn.dropout_keep_prob: 1.0  # KEEP 1 IN TESTING
                }
                step, loss, accuracy = sess.run(
                    # NO TRAIN_OP IN TESTING
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)

                dev_losses[batch_num] = loss
                dev_accuracies[batch_num] = accuracy

            print 'avg loss {} accuracy {}'.format(
                dev_losses.mean(), dev_accuracies.mean())
            # summaries, result1 = sess.run(
            #     [dev_summary_op, dev_loss],
            #     {dev_loss_batch: dev_losses,
            #      dev_accuracy_batch: dev_accuracies})
            # print ' dev loss ', result1
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, accuracy: {:g} ".format(
            # 	time_str, step, loss, accuracy))
            # if writer:
            #     writer.add_summary(summaries, step)

        for x_batch in train_batches:
            train_step(x_batch)
            current_step = tf.train.global_step(sess, global_step)
            # if current_step % FLAGS.evaluate_every == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_dev)  #, writer=dev_summary_writer)
            #     print("")
            if current_step % FLAGS.checkpoint_every == 0:
                # remember that Tensorflow variables are only alive
                # inside a session so => save the model inside a session
                path = saver.save(
                	sess, checkpoint_prefix, global_step=current_step,
                    write_meta_graph=savegraph
                )
                print("Saved model checkpoint to {}\n".format(path))
                savegraph = False
                # i.e. only first time save the graph since it doesnt change

                # to save a model every 2 hours and maximum 4 latest
                # models to be saved:
                # saver = tf.train.Saver(
                    # max_to_keep=4, keep_checkpoint_every_n_hours=2)

                # We can specify the variables/collections we want to save.
                # While creating the tf.train.Saver instance we pass it a list
                # or a dictionary of variables that we want to save:
                # saver = tf.train.Saver([w1,w2])
