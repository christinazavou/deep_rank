import os
import pickle
import gzip
import numpy as np
import tensorflow as tf
from prettytable import PrettyTable

from nn import get_activation_by_name
from qa import myio
from evaluation import Evaluation


class Model(object):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}

    def ready(self):
        self._initialize_graph()
        for param in tf.trainable_variables():
            self.params[param.name] = param

    def _initialize_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='target_tags')

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        with tf.name_scope('embeddings'):
            self.titles = tf.nn.embedding_lookup(self.embeddings, self.titles_words_ids_placeholder)
            self.bodies = tf.nn.embedding_lookup(self.embeddings, self.bodies_words_ids_placeholder)

            if self.weights is not None:
                titles_weights = tf.nn.embedding_lookup(self.weights, self.titles_words_ids_placeholder)
                titles_weights = tf.expand_dims(titles_weights, axis=2)
                self.titles = self.titles * titles_weights

                bodies_weights = tf.nn.embedding_lookup(self.weights, self.bodies_words_ids_placeholder)
                bodies_weights = tf.expand_dims(bodies_weights, axis=2)
                self.bodies = self.bodies * bodies_weights

            self.titles = tf.nn.dropout(self.titles, 1.0 - self.dropout_prob)
            self.bodies = tf.nn.dropout(self.bodies, 1.0 - self.dropout_prob)

        with tf.name_scope('CNN'):
            print 'ignoring depth at the moment !!'
            self.embedded_titles_expanded = tf.expand_dims(self.titles, -1)
            self.embedded_bodies_expanded = tf.expand_dims(self.bodies, -1)

            pooled_outputs_t = []
            pooled_outputs_b = []
            filter_sizes = [3]
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.embedding_layer.n_d, 1, self.args.hidden_dim]
                    print 'assuming num filters = hidden dim. IS IT CORRECT? '

                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv-W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.args.hidden_dim]), name="conv-b")

                    with tf.name_scope('titles_output'):
                        conv_t = tf.nn.conv2d(
                            self.embedded_titles_expanded,
                            W,
                            strides=[1, 1, 1, 1],  # how much the window shifts by in each of the dimensions.
                            padding="VALID",
                            name="conv-titles")

                        h_t = tf.nn.relu(tf.nn.bias_add(conv_t, b), name="relu-titles")

                        pooled_t = tf.reduce_max(
                            h_t,
                            axis=1,
                            keep_dims=True
                        )
                        pooled_outputs_t.append(pooled_t)

                    with tf.name_scope('bodies_output'):
                        conv_b = tf.nn.conv2d(
                            self.embedded_bodies_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv-bodies")

                        h_b = tf.nn.relu(tf.nn.bias_add(conv_b, b), name="relu-bodies")

                        pooled_b = tf.reduce_max(
                            h_b,
                            axis=1,
                            keep_dims=True
                        )
                        pooled_outputs_b.append(pooled_b)

            num_filters_total = self.args.hidden_dim * len(filter_sizes)
            self.t_pool = tf.concat(pooled_outputs_t, 3)
            self.t_state = tf.reshape(self.t_pool, [-1, num_filters_total])

            self.b_pool = tf.concat(pooled_outputs_b, 3)
            self.b_state = tf.reshape(self.b_pool, [-1, num_filters_total])

        with tf.name_scope('outputs'):

            with tf.name_scope('encodings'):
                # batch * d
                h_final = (self.t_state + self.b_state) * 0.5
                self.h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(self.h_final)

            with tf.name_scope("MLP"):
                self.w_o = tf.Variable(
                    tf.random_normal([self.args.hidden_dim, self.output_dim], mean=0.0, stddev=0.05),
                    name='weights_out'
                )
                self.b_o = tf.Variable(tf.zeros([self.output_dim]), name='bias_out')

            out = tf.matmul(self.h_final, self.w_o) + self.b_o
            self.output = tf.nn.sigmoid(out)

            # for evaluation
            self.prediction = tf.where(
                self.output > self.args.threshold, tf.ones_like(self.output), tf.zeros_like(self.output)
            )

            with tf.name_scope('cost'):

                with tf.name_scope('loss'):

                    if self.args.loss_type == 'xentropy':
                        self.loss = -tf.reduce_sum(
                            (self.target * tf.log(self.output + 1e-9)) + (
                                (1 - self.target) * tf.log(1 - self.output + 1e-9)),
                            name='cross_entropy'
                        )

                    else:
                        raise Exception('unimplemented')

                with tf.name_scope('regularization'):
                    l2_reg = 0.
                    for param in tf.trainable_variables():
                        l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                    self.l2_reg = l2_reg

                self.cost = self.loss + self.l2_reg

    @staticmethod
    def normalize_2d(x, eps=1e-8):
        # x is batch*hid_dim
        # l2 is batch*1
        l2 = tf.norm(x, ord=2, axis=1, keep_dims=True)
        return x / (l2 + eps)

    def eval_batch(self, titles, bodies, sess):
        outputs, predictions = sess.run(
            [self.output, self.prediction],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return outputs, predictions

    def evaluate(self, dev_batches, sess):
        outputs, predictions, targets = [], [], []
        for titles_b, bodies_b, tags_b in dev_batches:
            out, pred = self.eval_batch(titles_b, bodies_b, sess)
            outputs.append(out)
            predictions.append(pred)
            targets.append(tags_b)
        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object
        ev = Evaluation(outputs, predictions, targets)
        return ev.precision_recall_fscore('macro'), ev.precision_recall_fscore('micro')

    def train_batch(self, titles, bodies, y_batch, train_op, global_step, train_summary_op, train_summary_writer, sess):
        _, _step, _loss, _cost, _summary = sess.run(
            [train_op, global_step, self.loss, self.cost, train_summary_op],
            feed_dict={
                self.target: y_batch,
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: np.float32(self.args.dropout),
            }
        )
        train_summary_writer.add_summary(_summary, _step)
        return _step, _loss, _cost

    def train_model(self, train_batches, dev=None, test=None, assign_ops=None):
        with tf.Session() as sess:

            result_table = PrettyTable(
                ["Epoch", "dev A P", "dev A R", "dev A F1", "dev I P", "dev I R", "dev I F1",
                 "tst A P", "tst A R", "tst A F1", "tst I P", "tst I R", "tst I F1"]
            )

            dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1 = 0, 0, 0, 0, 0, 0
            test_MAC_P, test_MAC_R, test_MAC_F1, test_MIC_P, test_MIC_R, test_MIC_F1 = 0, 0, 0, 0, 0, 0

            best_dev_performance = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            grads_and_vars = optimizer.compute_gradients(self.cost)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            if assign_ops:
                print 'assigning trained values ...\n'
                sess.run(assign_ops)

            if self.args.save_dir != "":
                print("Writing to {}\n".format(self.args.save_dir))

            # Summaries for loss and cost
            loss_summary = tf.summary.scalar("loss", self.loss)
            cost_summary = tf.summary.scalar("cost", self.cost)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, cost_summary])
            train_summary_dir = os.path.join(self.args.save_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            if dev:
                # Dev Summaries
                dev_mac_p = tf.placeholder(tf.float32)
                dev_mic_p = tf.placeholder(tf.float32)
                dev_mac_r = tf.placeholder(tf.float32)
                dev_mic_r = tf.placeholder(tf.float32)
                dev_mac_f1 = tf.placeholder(tf.float32)
                dev_mic_f1 = tf.placeholder(tf.float32)
                dev_mac_p_summary = tf.summary.scalar("dev_mac_p", dev_mac_p)
                dev_mic_p_summary = tf.summary.scalar("dev_mic_p", dev_mic_p)
                dev_mac_r_summary = tf.summary.scalar("dev_mac_r", dev_mac_r)
                dev_mic_r_summary = tf.summary.scalar("dev_mic_r", dev_mic_r)
                dev_mac_f1_summary = tf.summary.scalar("dev_mac_f1", dev_mac_f1)
                dev_mic_f1_summary = tf.summary.scalar("dev_mic_f1", dev_mic_f1)
                dev_summary_op = tf.summary.merge(
                    [dev_mac_f1_summary, dev_mic_f1_summary,
                     dev_mac_p_summary, dev_mic_p_summary,
                     dev_mac_r_summary, dev_mic_r_summary]
                )
                dev_summary_dir = os.path.join(self.args.save_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            if test:
                # Test Summaries
                test_mac_p = tf.placeholder(tf.float32)
                test_mic_p = tf.placeholder(tf.float32)
                test_mac_r = tf.placeholder(tf.float32)
                test_mic_r = tf.placeholder(tf.float32)
                test_mac_f1 = tf.placeholder(tf.float32)
                test_mic_f1 = tf.placeholder(tf.float32)
                test_mac_p_summary = tf.summary.scalar("test_mac_p", test_mac_p)
                test_mic_p_summary = tf.summary.scalar("test_mic_p", test_mic_p)
                test_mac_r_summary = tf.summary.scalar("test_mac_r", test_mac_r)
                test_mic_r_summary = tf.summary.scalar("test_mic_r", test_mic_r)
                test_mac_f1_summary = tf.summary.scalar("test_mac_f1", test_mac_f1)
                test_mic_f1_summary = tf.summary.scalar("test_mic_f1", test_mic_f1)
                test_summary_op = tf.summary.merge(
                    [test_mac_f1_summary, test_mic_f1_summary,
                     test_mac_p_summary, test_mic_p_summary,
                     test_mac_r_summary, test_mic_r_summary]
                )
                test_summary_dir = os.path.join(self.args.save_dir, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            if self.args.save_dir != "":
                checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

            unchanged = 0
            max_epoch = 50
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > 15:
                    break

                N = len(train_batches)

                train_loss = 0.0
                train_cost = 0.0

                for i in xrange(N):
                    titles_b, bodies_b, tag_labels_b = train_batches[i]
                    cur_step, cur_loss, cur_cost = self.train_batch(
                        titles_b, bodies_b, tag_labels_b,
                        train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))

                    if i == N-1:  # EVAL
                        if dev:
                            (dev_MAC_P, dev_MAC_P, dev_MAC_F1), (dev_MIC_P, dev_MIC_R, dev_MIC_F1) = \
                                self.evaluate(dev, sess)
                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_mac_f1: dev_MAC_F1, dev_mic_f1: dev_MIC_F1,
                                 dev_mac_p: dev_MAC_P, dev_mic_p: dev_MIC_P,
                                 dev_mac_r: dev_MAC_R, dev_mic_r: dev_MIC_R}
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            (test_MAC_P, test_MAC_P, test_MAC_F1), (test_MIC_P, test_MIC_R, test_MIC_F1) = \
                                self.evaluate(test, sess)
                            _test_sum = sess.run(
                                test_summary_op,
                                {test_mac_f1: test_MAC_F1, test_mic_f1: test_MIC_F1,
                                 test_mac_p: test_MAC_P, test_mic_p: test_MIC_P,
                                 test_mac_r: test_MAC_R, test_mic_r: test_MIC_R}
                            )
                            test_summary_writer.add_summary(_test_sum, cur_step)

                        # if dev_MIC_P > best_dev_performance:
                        if dev_MIC_F1 > best_dev_performance:
                            unchanged = 0
                            # best_dev_performance = dev_MIC_P
                            best_dev_performance = dev_MIC_F1
                            result_table.add_row(
                                [epoch, dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1,
                                 test_MAC_P, test_MAC_R, test_MAC_F1, test_MIC_P, test_MIC_R, test_MIC_F1]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        myio.say("\r\n\nEpoch {}\tcost={:.3f}\tloss={:.3f}\tPRE={:.2f},{:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss / (i+1),  # i.e. divided by N training batches
                            dev_MIC_P,
                            best_dev_performance
                        ))
                        myio.say("\n{}\n".format(result_table))

    def save(self, sess, path, step):
        path = "{}_{}_{}".format(path, step, ".pkl.gz")
        print("Saving model checkpoint to {}\n".format(path))
        params_values = {}
        for param_name, param in self.params.iteritems():
            params_values[param.name] = sess.run(param)
        # print 'params_dict\n', params_dict
        with gzip.open(path, "w") as fout:
            pickle.dump(
                {
                    "params_values": params_values,
                    "args": self.args,
                    "step": step
                },
                fout,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def load_trained_vars(self, path):
        print("Loading model checkpoint from {}\n".format(path))
        assert self.args is not None and self.params != {}
        assign_ops = {}
        with gzip.open(path) as fin:
            data = pickle.load(fin)
            assert self.args.hidden_dim == data["args"].hidden_dim
            params_values = data['params_values']
            graph = tf.get_default_graph()
            for param_name, param_value in params_values.iteritems():
                variable = graph.get_tensor_by_name(param_name)
                assign_op = tf.assign(variable, param_value)
                assign_ops[param_name] = assign_op
        return assign_ops

    def load_n_set_model(self, path, sess):
        with gzip.open(path) as fin:
            data = pickle.load(fin)
        self.args = data["args"]
        self.ready()
        assign_ops = self.load_trained_vars(path)
        sess.run(tf.global_variables_initializer())
        print 'assigning trained values ...\n'
        for param_name, param_assign_op in assign_ops.iteritems():
            sess.run(param_assign_op)

    def num_parameters(self):
        total_parameters = 0
        for param_name, param in self.params.iteritems():
            # shape is an array of tf.Dimension
            shape = param.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
