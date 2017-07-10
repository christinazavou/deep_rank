# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import time
import numpy as np
import datetime
from askubuntu.main import triples
from askubuntu.deepranker import TextCNN
from askubuntu.cnn_config import FLAGS
from askubuntu.data_helpers import batch_iter


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

x_train, x_dev = triples()
train_batches = batch_iter(
    x_train,
    FLAGS.batch_size,
    FLAGS.num_epochs,
    shuffle=True
)

# metagraphfile = os.path.join(
#     this_dir.replace('askubuntu', 'askubunturuns'),
#     '1499423611', 'checkpoints', 'model-5400.meta'
# )
# embeddingW = None
# convmaxpool2W, convmaxpool3W, convmaxpool4W = None, None, None
# convmaxpool2b, convmaxpool3b, convmaxpool4b = None, None, None
# if os.path.isfile(metagraphfile):
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph(metagraphfile)
#         saver.restore(sess, metagraphfile.replace('.meta', ''))
#         graph = tf.get_default_graph()

#         embeddingW = graph.get_tensor_by_name('embedding/W:0')
#         convmaxpool2W = graph.get_tensor_by_name('conv-maxpool-2/W:0')
#         convmaxpool3W = graph.get_tensor_by_name('conv-maxpool-3/W:0')
#         convmaxpool4W = graph.get_tensor_by_name('conv-maxpool-4/W:0')
#         convmaxpool2b = graph.get_tensor_by_name('conv-maxpool-2/b:0')
#         convmaxpool3b = graph.get_tensor_by_name('conv-maxpool-3/b:0')
#         convmaxpool4b = graph.get_tensor_by_name('conv-maxpool-4/b:0')


# metagraphfile = os.path.join(
#     this_dir.replace('askubuntu', 'askubunturuns'),
#     '1499423611', 'checkpoints', 'model-5400.meta'
# )
# if os.path.isfile(metagraphfile):
#     with tf.Session() as sess:
#         # create the network
#         saver = tf.train.import_meta_graph(metagraphfile)
#         # import_meta_graph appends the network defined in .meta file
#         # to the current graph => creates the graph/network for you but
#         # we still need to load the value of the parameters that we had
#         # trained on this graph.

#         # load the parameters
#         saver.restore(sess, metagraphfile.replace('.meta', ''))
#         # # Access saved Variables directly
#         graph = tf.get_default_graph()
#         # print [n.name for n in graph.as_graph_def().node]
#         # print graph.get_tensor_by_name('conv-maxpool-4/b:0')
#         # note: without ":0" it identifies operation
#         x = graph.get_tensor_by_name('input_x:0')
#         x1 = graph.get_tensor_by_name('input_x1:0')
#         x2 = graph.get_tensor_by_name('input_x2:0')
#         dropout = graph.get_tensor_by_name('dropout_keep_prob:0')
#         accuracy = graph.get_operation_by_name('accuracy/accuracy')

#         for x_batch in train_batches:
#             feed_dict = {
#                 x: np.asarray([xi[0] for xi in x_batch]),
#                 x1: np.asarray([xi[1] for xi in x_batch]),
#                 x2: np.asarray([xi[2] for xi in x_batch]),
#                 dropout: 1.0
#             }
#             acc = sess.run([accuracy], feed_dict)
#             print("accuracy: {} ".format(acc))
#             break

with tf.Graph().as_default():
    savegraph = True
    session_conf = tf.ConfigProto(
    	allow_soft_placement=FLAGS.allow_soft_placement,
    	log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        cnn = TextCNN(
            sequence_length=FLAGS.sequence_length,
            num_classes=1,
            vocab_size=FLAGS.vocabulary_size + 1,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
        	grads_and_vars, global_step=global_step)

        print 'total params: ', total_parameters()
        exit()

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
        	os.path.join(os.path.curdir, FLAGS.out_dir, timestamp))
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
            # similarity1, similarity2, similarities, predictions, \
            #     correct_predictions =\
            #     sess.run(
            #         [cnn.similarity1, cnn.similarity2, cnn.similarities,
            #          cnn.predictions, cnn.correct_predictions],
            #         feed_dict
            #     )
            # print ' similarity 1 ', similarity1, ' \n'
            # print ' similarity 2 ', similarity2, ' \n'
            # print ' similarities ', similarities, ' \n'
            # print ' predictions ', predictions, ' \n'
            # print ' correct predictions ', correct_predictions, ' \n'
            # exit()

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
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev)  #, writer=dev_summary_writer)
                print("")
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
