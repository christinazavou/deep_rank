import tensorflow as tf
from nn import get_activation_by_name
from qa.evaluation import Evaluation
import numpy as np
import os
import gzip
import pickle
from prettytable import PrettyTable
from qa.myio import say
from tensorflow.contrib.rnn import LSTMStateTuple


def bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs_embedded,
        input_lengths,
        scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs_embedded,
            sequence_length=input_lengths,
            dtype=tf.float32,
            swap_memory=True,
            scope=scope
        )
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1, name='bidirectional_concat')
                return state
            elif isinstance(fw_state, tuple) and isinstance(bw_state, tuple) and (len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state))
                return state
            else:
                raise ValueError('unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)

        return outputs, state


class HANClassifierModel(object):

    def __init__(self, args, embedding_layer, word_hid_dim, sent_hid_dim, word_attention_size,
                 sent_attention_size, word_sequence_length, sent_sequence_length):

        self.args = args
        self.embeddings = embedding_layer.embeddings
        self.embedding_size = embedding_layer.n_d
        self.padding_id = embedding_layer.vocab_map["<padding>"]

        self.word_attention_size = word_attention_size
        self.word_hid_dim = word_hid_dim
        self.word_sequence_length = word_sequence_length

        self.sent_attention_size = sent_attention_size
        self.sent_hid_dim = sent_hid_dim
        self.sent_sequence_length = sent_sequence_length

        self.params = {}

    def ready(self):
        self._initialize_placeholders_graph()
        self._initialize_encoder_graph()
        self._initialize_output_graph()
        for param in tf.trainable_variables():
            self.params[param.name] = param

    def _initialize_placeholders_graph(self):

        with tf.name_scope('input'):
            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def _initialize_encoder_graph(self):
        with tf.variable_scope('embeddings'):

            # [document x sentence], [document]
            self.word_lengths, self.sent_lengths = self._find_sequence_lengths(self.inputs)
            self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs)

            # self.inputs_embedded = tf.nn.dropout(self.inputs_embedded, 1.0 - self.dropout_prob)

        with tf.variable_scope('HierarchicalAttention'):

            """ -------------------------------------------WORD LEVEL----------------------------------------------- """

            word_level_inputs = tf.reshape(
                self.inputs_embedded,
                [-1, self.word_sequence_length, self.embedding_size]
            )
            self.word_level_inputs = word_level_inputs

            word_level_lengths = tf.reshape(self.word_lengths, [-1])

            def lstm_cell(state_size):
                _cell = tf.nn.rnn_cell.LSTMCell(
                    state_size, state_is_tuple=True, activation=get_activation_by_name(self.args.activation)
                )
                return _cell

            def gru_cell(state_size):
                _cell = tf.nn.rnn_cell.GRUCell(
                    state_size, activation=get_activation_by_name(self.args.activation)
                )
                return _cell

            if self.args.layer.lower() == "lstm":
                word_cell = lstm_cell(self.word_hid_dim)
            else:
                word_cell = gru_cell(self.word_hid_dim)

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    word_cell,
                    word_cell,
                    word_level_inputs,
                    word_level_lengths,
                    scope=scope
                )

                self.word_hid_dim = self.word_hid_dim * 2  # DUE TO BIDIRECTIONAL

                self.word_encoder_output = word_encoder_output

                with tf.variable_scope('attention'):

                    w_omega_w = tf.Variable(tf.random_normal([self.word_hid_dim, self.word_attention_size], stddev=0.1))
                    b_omega_w = tf.Variable(tf.random_normal([self.word_attention_size], stddev=0.1))
                    u_omega_w = tf.Variable(tf.random_normal([self.word_attention_size], stddev=0.1))

                    v_w = tf.tanh(
                        tf.matmul(tf.reshape(word_encoder_output, [-1, self.word_hid_dim]), w_omega_w)
                        +
                        tf.reshape(b_omega_w, [1, -1])
                    )
                    vu_w = tf.matmul(v_w, tf.reshape(u_omega_w, [-1, 1]))
                    exps_w = tf.reshape(tf.exp(vu_w), [-1, self.word_sequence_length])

                    alphas_w = exps_w / tf.reshape(tf.reduce_sum(exps_w, 1), [-1, 1])

                    # Output of previous layer (i.e. the encodings given here) reduced with the attention vector
                    word_level_output = tf.reduce_sum(
                        word_encoder_output * tf.reshape(alphas_w, [-1, self.word_sequence_length, 1]),
                        1
                    )
                    self.word_level_output = word_level_output

                with tf.variable_scope('dropout'):
                    word_level_output = tf.nn.dropout(word_level_output, 1.0 - self.dropout_prob)

            """ ---------------------------------------SENTENCE LEVEL----------------------------------------------- """

            sent_level_inputs = tf.reshape(
                word_level_output,
                [-1, self.sent_sequence_length, self.word_hid_dim]
            )
            self.sent_level_inputs = sent_level_inputs

            if self.args.layer.lower() == "lstm":
                sent_cell = lstm_cell(self.sent_hid_dim)
            else:
                sent_cell = gru_cell(self.sent_hid_dim)

            with tf.variable_scope('sentence') as scope:
                sent_encoder_output, _ = bidirectional_rnn(
                    sent_cell,
                    sent_cell,
                    sent_level_inputs,
                    self.sent_lengths,
                    scope=scope
                )

                self.sent_hid_dim = self.sent_hid_dim * 2  # DUE TO BIDIRECTIONAL

                self.sent_encoder_output = sent_encoder_output

                with tf.variable_scope('attention'):

                    w_omega_s = tf.Variable(tf.random_normal([self.sent_hid_dim, self.sent_attention_size], stddev=0.1))
                    b_omega_s = tf.Variable(tf.random_normal([self.sent_attention_size], stddev=0.1))
                    u_omega_s = tf.Variable(tf.random_normal([self.sent_attention_size], stddev=0.1))

                    v_s = tf.tanh(
                        tf.matmul(tf.reshape(sent_encoder_output, [-1, self.sent_hid_dim]), w_omega_s)
                        +
                        tf.reshape(b_omega_s, [1, -1])
                    )
                    vu_s = tf.matmul(v_s, tf.reshape(u_omega_s, [-1, 1]))
                    exps_s = tf.reshape(tf.exp(vu_s), [-1, self.sent_sequence_length])

                    alphas_s = exps_s / tf.reshape(tf.reduce_sum(exps_s, 1), [-1, 1])

                    # Output of previous layer (i.e. the encodings given here) reduced with the attention vector
                    sent_level_output = tf.reduce_sum(
                        sent_encoder_output * tf.reshape(alphas_s, [-1, self.sent_sequence_length, 1]),
                        1
                    )
                    self.sent_level_output = sent_level_output

                with tf.variable_scope('dropout'):
                    sent_level_output = tf.nn.dropout(sent_level_output, 1.0 - self.dropout_prob)

            with tf.name_scope('outputs'):
                # batch * d
                h_final = sent_level_output
                assert len(h_final.get_shape()) == 2
                self.h_final = self.normalize_2d(h_final)

    def _initialize_output_graph(self):

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

    @staticmethod
    def normalize_2d(x, eps=1e-8):
        # x is batch*hid_dim
        # l2 is batch*1
        l2 = tf.norm(x, ord=2, axis=1, keep_dims=True)
        return x / (l2 + eps)

    def _find_sequence_lengths(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        wl = tf.reduce_sum(s, axis=2)
        s = tf.not_equal(wl, self.padding_id)
        s = tf.cast(s, tf.int32)
        sl = tf.reduce_sum(s, axis=1)
        assert len(wl.get_shape()) == 2 and len(sl.get_shape()) == 1
        return wl, sl

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

    def eval_batch(self, questions, sess):
        _scores = sess.run(
            self.scores,
            feed_dict={
                self.inputs: questions,
                self.dropout_prob: 0.,
            }
        )
        return _scores

    def evaluate(self, data, sess):
        res = []
        hinge_loss = 0.

        for questions, id_labels in data:
            cur_scores = self.eval_batch(questions, sess)
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

    def train_batch(self, questions, pairs, train_op, global_step, train_summary_op, train_summary_writer, sess):
        _, _step, _loss, _cost, _summary = sess.run(
            [train_op, global_step, self.loss, self.cost, train_summary_op],
            feed_dict={
                self.inputs: questions,
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
                del assign_ops

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
                    questions, idps = train_batches[i]
                    cur_step, cur_loss, cur_cost = self.train_batch(
                        questions, idps, train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        say("\r{}/{}".format(i, N))

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
            assert self.args.sent_hid_dim == data["args"].sent_hid_dim
            assert self.args.word_hid_dim == data["args"].word_hid_dim
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
            assert self.args.sent_hid_dim == data["args"].sent_hid_dim
            assert self.args.word_hid_dim == data["args"].word_hid_dim
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
