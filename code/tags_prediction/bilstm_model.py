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

        self.SLT = self._find_sequence_length(self.titles_words_ids_placeholder)
        self.SLB = self._find_sequence_length(self.bodies_words_ids_placeholder)

        with tf.name_scope('embeddings'):
            self.titles = tf.nn.embedding_lookup(self.embeddings, self.titles_words_ids_placeholder)
            self.bodies = tf.nn.embedding_lookup(self.embeddings, self.bodies_words_ids_placeholder)

            if self.weights is not None:
                print 'weighting the embeddings...'
                titles_weights = tf.nn.embedding_lookup(self.weights, self.titles_words_ids_placeholder)
                titles_weights = tf.expand_dims(titles_weights, axis=2)
                self.titles = self.titles * titles_weights

                bodies_weights = tf.nn.embedding_lookup(self.weights, self.bodies_words_ids_placeholder)
                bodies_weights = tf.expand_dims(bodies_weights, axis=2)
                self.bodies = self.bodies * bodies_weights

            self.titles = tf.nn.dropout(self.titles, 1.0 - self.dropout_prob)
            self.bodies = tf.nn.dropout(self.bodies, 1.0 - self.dropout_prob)

        with tf.name_scope('LSTM'):

            def lstm_cell(state_size):
                _cell = tf.nn.rnn_cell.LSTMCell(
                    state_size, state_is_tuple=True, activation=get_activation_by_name(self.args.activation)
                )
                # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                return _cell

            forward_cell = lstm_cell(self.args.hidden_dim/2 if self.args.average == 0 else self.args.hidden_dim)
            backward_cell = lstm_cell(self.args.hidden_dim/2 if self.args.average == 0 else self.args.hidden_dim)

        with tf.name_scope('titles_output'):
            self.t_outputs, self.t_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=self.titles,
                dtype=tf.float32,
                sequence_length=self.SLT
            )
            forw_t_state, back_t_state = self.t_state

            if self.args.average == 0:
                self.t_state_vec = tf.concat([forw_t_state[1], back_t_state[1]], axis=1)
            else:
                self.t_state_vec = (forw_t_state[1] + back_t_state[1]) / 2.

        with tf.name_scope('bodies_output'):
            self.b_outputs, self.b_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )

            forw_b_state, back_b_state = self.b_state

            if self.args.average == 0:
                self.b_state_vec = tf.concat([forw_b_state[1], back_b_state[1]], axis=1)
            else:
                self.b_state_vec = (forw_b_state[1] + back_b_state[1]) * 0.5

        with tf.name_scope('outputs'):

            with tf.name_scope('encodings'):
                # batch * d
                h_final = (self.t_state_vec + self.b_state_vec) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)
                # todo: can add dropout before inputting to MLP. normalization can be removed. ?!

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

                    self.loss = -tf.reduce_sum(
                        (self.target * tf.log(self.output + 1e-9)) + (
                        (1 - self.target) * tf.log(1 - self.output + 1e-9)),
                        name='cross_entropy'
                    )

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

    @staticmethod
    def normalize_3d(x, eps=1e-8):
        # x is len*batch*hid_dim
        # l2 is len*batch*1
        l2 = tf.norm(x, ord=2, axis=2, keep_dims=True)
        return x / (l2 + eps)

    def average_without_padding(self, x, ids, eps=1e-8):
        # len*batch*1
        mask = tf.not_equal(ids, self.padding_id)
        mask = tf.expand_dims(mask, 2)
        mask = tf.cast(mask, tf.float32)
        # batch*d
        s = tf.reduce_sum(x*mask, axis=1) / (tf.reduce_sum(mask, axis=1)+eps)
        return s

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

        # ev = Evaluation(outputs, predictions, y_batch)
        # mac_p, mac_r, mac_f1 = ev.precision_recall_fscore(average='macro')
        # mic_p, mic_r, mic_f1 = ev.precision_recall_fscore(average='micro')
        # # oe = ev.one_error()
        # return (mac_p, mac_r, mac_f1), (mic_p, mic_r, mic_f1)

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

            # dev_OE, dev_HL, dev_BP_MLL = 0
            dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1 = 0, 0, 0, 0, 0, 0
            # test_OE, test_HL, test_BP_MLL = 0
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

            # Train Summaries
            loss_summary = tf.summary.scalar("loss", self.loss)
            cost_summary = tf.summary.scalar("cost", self.cost)
            train_summary_op = tf.summary.merge([loss_summary, cost_summary])
            train_summary_dir = os.path.join(self.args.save_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            if dev:
                # Dev Summaries
                dev_mac_p = tf.placeholder(tf.float32)
                dev_mic_p = tf.placeholder(tf.float32)
                dev_mac_p_summary = tf.summary.scalar("dev_mac_p", dev_mac_p)
                dev_mic_p_summary = tf.summary.scalar("dev_mic_p", dev_mic_p)

                dev_mac_r = tf.placeholder(tf.float32)
                dev_mic_r = tf.placeholder(tf.float32)
                dev_mac_r_summary = tf.summary.scalar("dev_mac_r", dev_mac_r)
                dev_mic_r_summary = tf.summary.scalar("dev_mic_r", dev_mic_r)

                dev_mac_f1 = tf.placeholder(tf.float32)
                dev_mic_f1 = tf.placeholder(tf.float32)
                dev_mac_f1_summary = tf.summary.scalar("dev_mac_f1", dev_mac_f1)
                dev_mic_f1_summary = tf.summary.scalar("dev_mic_f1", dev_mic_f1)

                dev_summary_op = tf.summary.merge(
                    [dev_mac_p_summary, dev_mic_p_summary,
                     dev_mac_r_summary, dev_mic_r_summary,
                     dev_mac_f1_summary, dev_mic_f1_summary]
                )
                dev_summary_dir = os.path.join(self.args.save_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            if test:
                # Test Summaries
                test_mac_p = tf.placeholder(tf.float32)
                test_mic_p = tf.placeholder(tf.float32)
                test_mac_p_summary = tf.summary.scalar("test_mac_p", test_mac_p)
                test_mic_p_summary = tf.summary.scalar("test_mic_p", test_mic_p)

                test_mac_r = tf.placeholder(tf.float32)
                test_mic_r = tf.placeholder(tf.float32)
                test_mac_r_summary = tf.summary.scalar("test_mac_r", test_mac_r)
                test_mic_r_summary = tf.summary.scalar("test_mic_r", test_mic_r)

                test_mac_f1 = tf.placeholder(tf.float32)
                test_mic_f1 = tf.placeholder(tf.float32)
                test_mac_f1_summary = tf.summary.scalar("test_mac_f1", test_mac_f1)
                test_mic_f1_summary = tf.summary.scalar("test_mic_f1", test_mic_f1)

                test_summary_op = tf.summary.merge(
                    [test_mac_p_summary, test_mic_p_summary,
                     test_mac_r_summary, test_mic_r_summary,
                     test_mac_f1_summary, test_mic_f1_summary]
                )
                test_summary_dir = os.path.join(self.args.save_dir, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            if self.args.save_dir != "":
                checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

            unchanged = 0
            max_epoch = self.args.max_epoch
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
                            (dev_MAC_P, dev_MAC_R, dev_MAC_F1), (dev_MIC_P, dev_MIC_R, dev_MIC_F1) = \
                                self.evaluate(dev, sess)
                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_mac_f1: dev_MAC_F1, dev_mic_f1: dev_MIC_F1,
                                 dev_mac_p: dev_MAC_P, dev_mic_p: dev_MIC_P,
                                 dev_mac_r: dev_MAC_R, dev_mic_r: dev_MIC_R}
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            (test_MAC_P, test_MAC_R, test_MAC_F1), (test_MIC_P, test_MIC_R, test_MIC_F1) = \
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
        # NOTE: Optimizer is not saved!!! So if more train..optimizer starts again
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

    def load_pre_trained_part(self, path):
        print("Loading model checkpoint from {}\n".format(path))
        assert self.args is not None and self.params != {}
        assign_ops = {}
        with gzip.open(path) as fin:
            data = pickle.load(fin)
            assert self.args.hidden_dim == data["args"].hidden_dim
            params_values = data['params_values']
            graph = tf.get_default_graph()
            for param_name, param_value in params_values.iteritems():
                if param_name in self.params:
                    print param_name, ' is in my dict'
                    try:
                        variable = graph.get_tensor_by_name(param_name)
                        assign_op = tf.assign(variable, param_value)
                        assign_ops[param_name] = assign_op
                    except:
                        raise Exception("{} not found in my graph".format(param_name))
                else:
                    print param_name, ' is not in my dict'
        return assign_ops

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

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s

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
