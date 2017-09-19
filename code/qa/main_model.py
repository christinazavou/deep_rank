import tensorflow as tf
import numpy as np
from evaluation import Evaluation
from nn import get_activation_by_name
import gzip
import pickle
from prettytable import PrettyTable
from myio import say
import myio
import os


class Model(object):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.params = {}

    def ready(self):
        self._initialize_graph()
        for param in tf.trainable_variables():
            self.params[param.name] = param

    def _initialize_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

        self.SLT = self._find_sequence_length(self.titles_words_ids_placeholder)
        self.SLB = self._find_sequence_length(self.bodies_words_ids_placeholder)

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

        with tf.name_scope('LSTM'):

            def lstm_cell():
                _cell = tf.nn.rnn_cell.LSTMCell(
                    self.args.hidden_dim, state_is_tuple=True, activation=get_activation_by_name(self.args.activation)
                )
                # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                return _cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(self.args.depth)]
            )

        with tf.name_scope('titles_output'):
            self.t_states_series, self.t_current_state = tf.nn.dynamic_rnn(
                cell,
                self.titles,
                dtype=tf.float32,
                sequence_length=self.SLT
            )
            # current_state = last state of every layer in the network as an LSTMStateTuple

            if self.args.normalize:
                self.t_states_series = self.normalize_3d(self.t_states_series)

            if self.args.average:
                self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
            else:
                # self.t_state=self.t_states_series[:, -1, :]=self.t_current_state[-1][1]=self.t_current_state[0][1]
                # in case sequence_length parameter is used in RNN, the last state is not self.t_states_series[:,-1,:]
                # but is self.t_states_series[:, self.SLT[x], :] and it is stored correctly in
                # self.t_current_state[0][1] so its better and safer to use this.
                self.t_state = self.t_current_state[0][1]

        with tf.name_scope('bodies_output'):
            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                cell,
                self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )
            # current_state = last state of every layer in the network as an LSTMStateTuple

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            else:
                # self.b_state=self.b_states_series[:, -1, :]=self.b_current_state[-1][1]=self.b_current_state[0][1]
                # in case sequence_length parameter is used in RNN, the last state is not self.b_states_series[:,-1,:]
                # but is self.b_states_series[:, self.SLB[x], :] and it is stored correctly in
                # self.b_current_state[0][1] so its better and safer to use this.
                self.b_state = self.b_current_state[0][1]

        with tf.name_scope('outputs'):
            # batch * d
            h_final = (self.t_state + self.b_state) * 0.5
            h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
            self.h_final = self.normalize_2d(h_final)

        with tf.name_scope('scores'):
            # For testing:
            #   first one in batch is query, the rest are candidate questions
            self.scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)

            # For training:
            pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
            # num query * n_d
            query_vecs = pairs_vecs[:, 0, :]
            # num query
            pos_scores = tf.reduce_sum(query_vecs * pairs_vecs[:, 1, :], axis=1)
            # num query * candidate size
            neg_scores = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 2:, :], axis=2)
            # num query
            neg_scores = tf.reduce_max(neg_scores, axis=1)

        with tf.name_scope('cost'):

            with tf.name_scope('loss'):
                diff = neg_scores - pos_scores + 1.0
                loss = tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff)
                self.loss = loss

            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in tf.trainable_variables():
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg

            self.cost = self.loss + self.l2_reg

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s

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

    @staticmethod
    def max_margin_loss(labels, scores):
        pos_scores = [score for label, score in zip(labels, scores) if label == 1]
        neg_scores = [score for label, score in zip(labels, scores) if label == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return None
        pos_score = min(pos_scores)
        neg_score = max(neg_scores)
        diff = neg_score - pos_score + 1.0
        if diff > 0:
            loss = diff
        else:
            loss = 0
        return loss

    def eval_batch(self, titles, bodies, sess):
        _scores = sess.run(
            self.scores,
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return _scores

    def evaluate(self, data, sess):
        res = []
        hinge_loss = 0.

        for idts, idbs, id_labels in data:
            cur_scores = self.eval_batch(idts, idbs, sess)
            mml = self.max_margin_loss(id_labels, cur_scores)
            if mml is not None:
                hinge_loss = (hinge_loss + mml) / 2.
            assert len(id_labels) == len(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = Evaluation(res)
        MAP = round(e.MAP(), 4)
        MRR = round(e.MRR(), 4)
        P1 = round(e.Precision(1), 4)
        P5 = round(e.Precision(5), 4)
        return MAP, MRR, P1, P5, hinge_loss

    def train_batch(self, titles, bodies, pairs, train_op, global_step, train_summary_op, train_summary_writer, sess):
        _, _step, _loss, _cost, _summary = sess.run(
            [train_op, global_step, self.loss, self.cost, train_summary_op],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.dropout_prob: np.float32(self.args.dropout),
            }
        )
        train_summary_writer.add_summary(_summary, _step)
        return _step, _loss, _cost

    def train_model(self, train_batches, dev=None, test=None, assign_ops=None):
        with tf.Session() as sess:

            result_table = PrettyTable(
                ["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5", "tst MAP", "tst MRR", "tst P@1", "tst P@5"]
            )
            dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
            test_MAP = test_MRR = test_P1 = test_P5 = 0
            best_dev = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
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

            # Dev Summaries
            dev_loss = tf.placeholder(tf.float32)
            dev_map = tf.placeholder(tf.float32)
            dev_mrr = tf.placeholder(tf.float32)
            dev_loss_summary = tf.summary.scalar("dev_loss", dev_loss)
            dev_map_summary = tf.summary.scalar("dev_map", dev_map)
            dev_mrr_summary = tf.summary.scalar("dev_mrr", dev_mrr)
            dev_summary_op = tf.summary.merge([dev_loss_summary, dev_map_summary, dev_mrr_summary])
            dev_summary_dir = os.path.join(self.args.save_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

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
                    idts, idbs, idps = train_batches[i]
                    cur_step, cur_loss, cur_cost = self.train_batch(
                        idts, idbs, idps, train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))

                    if i == N-1:  # EVAL
                        if dev:
                            dev_MAP, dev_MRR, dev_P1, dev_P5, dev_hinge_loss = self.evaluate(dev, sess)
                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_loss: dev_hinge_loss, dev_map: dev_MAP, dev_mrr: dev_MRR}
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            test_MAP, test_MRR, test_P1, test_P5, test_hinge_loss = self.evaluate(test, sess)

                        if dev_MRR > best_dev:
                            unchanged = 0
                            best_dev = dev_MRR
                            result_table.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        say("\r\n\nEpoch {}\tcost={:.3f}\tloss={:.3f}\tMRR={:.2f},{:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss / (i+1),  # i.e. divided by N training batches
                            dev_MRR,
                            best_dev
                        ))
                        say("\n{}\n".format(result_table))

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

