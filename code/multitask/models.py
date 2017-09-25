import tensorflow as tf
import numpy as np
from qa.evaluation import Evaluation as QAEvaluation
from tags_prediction.evaluation import Evaluation as TPEvaluation
from nn import get_activation_by_name
import gzip
import pickle
from prettytable import PrettyTable
from qa.myio import say
import os


class ModelQATP(object):

    def ready(self):
        self._initialize_placeholders_graph()
        self._initialize_encoder_graph()
        self._initialize_output_graph_qa()
        self._initialize_output_graph_tp()
        for param in tf.trainable_variables():
            self.params[param.name] = param
        self._initialize_cost_function()

    def _initialize_placeholders_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22
            self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='target_tags')

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

    def _initialize_output_graph_qa(self):

        with tf.name_scope('scores'):
            self.scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)

            pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
            query_vecs = pairs_vecs[:, 0, :]
            pos_scores = tf.reduce_sum(query_vecs * pairs_vecs[:, 1, :], axis=1)
            neg_scores = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 2:, :], axis=2)
            neg_scores = tf.reduce_max(neg_scores, axis=1)

        with tf.name_scope('QaLoss'):
            diff = neg_scores - pos_scores + 1.0
            loss = tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff)
            self.loss_qa = loss

    def _initialize_output_graph_tp(self):

        with tf.name_scope('outputs'):

            with tf.name_scope("MLP"):
                self.w_o = tf.Variable(
                    tf.random_normal([self.args.hidden_dim, self.output_dim], mean=0.0, stddev=0.05),
                    name='weights_out'
                )
                self.b_o = tf.Variable(tf.zeros([self.output_dim]), name='bias_out')

            out = tf.matmul(self.h_final, self.w_o) + self.b_o
            self.output = tf.nn.sigmoid(out)

            self.prediction = tf.where(
                self.output > self.args.threshold, tf.ones_like(self.output), tf.zeros_like(self.output)
            )

        with tf.name_scope('TpLoss'):
            self.loss_tp = -tf.reduce_sum(
                (self.target * tf.log(self.output + 1e-9)) + ((1 - self.target) * tf.log(1 - self.output + 1e-9)),
                name='cross_entropy'
            )

    def _initialize_cost_function(self):
        with tf.name_scope('cost'):
            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in tf.trainable_variables():
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg
            self.cost = self.loss_qa + self.loss_tp + self.l2_reg

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
        scores, outputs, predictions = sess.run(
            [self.scores, self.output, self.prediction],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return scores, outputs, predictions

    def evaluate(self, data, sess):
        res = []
        hinge_loss = 0.

        outputs, predictions, targets = [], [], []

        for idts, idbs, id_labels, tags_b in data:
            cur_scores, cur_out, cur_pred = self.eval_batch(idts, idbs, sess)

            outputs.append(cur_out)
            predictions.append(cur_pred)
            targets.append(tags_b)

            mml = self.max_margin_loss(id_labels, cur_scores)
            if mml is not None:
                hinge_loss = (hinge_loss + mml) / 2.
            assert len(id_labels) == len(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = QAEvaluation(res)
        MAP = round(e.MAP(), 4)
        MRR = round(e.MRR(), 4)
        P1 = round(e.Precision(1), 4)
        P5 = round(e.Precision(5), 4)

        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object
        ev = TPEvaluation(outputs, predictions, targets)
        results = [round(ev.lr_ap_score(), 4), round(ev.lr_loss(), 4), round(ev.cov_error(), 4)]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        eval_labels = []
        for label in range(targets.shape[1]):
            if (targets[:, label] == np.ones(targets.shape[0])).any():
                eval_labels.append(label)
        print '\n{} labels out of {} will be evaluated (zero-sampled-labels removed).'.format(len(eval_labels), targets.shape[1])
        outputs, predictions, targets = outputs[:, eval_labels], predictions[:, eval_labels], targets[:, eval_labels]

        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, predictions, targets = outputs[eval_samples, :], predictions[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        ev = TPEvaluation(outputs, predictions, targets)
        results += [ev.precision_recall_fscore('macro'), ev.precision_recall_fscore('micro')]

        return MAP, MRR, P1, P5, hinge_loss, tuple(results)

    def train_batch(self, batch, train_op, global_step, train_summary_op, train_summary_writer, sess):
        titles, bodies, pairs, tags = batch
        _, _step, _loss_qa, _loss_tp, _cost, _summary = sess.run(
            [train_op, global_step, self.loss_qa, self.loss_tp, self.cost, train_summary_op],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.target: tags,
                self.dropout_prob: np.float32(self.args.dropout),
            }
        )
        train_summary_writer.add_summary(_summary, _step)
        return _step, _loss_qa, _loss_tp, _cost

    def train_model(self, train_batches, dev=None, test=None, assign_ops=None):
        with tf.Session() as sess:

            result_table_qa = PrettyTable(
                ["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5", "tst MAP", "tst MRR", "tst P@1", "tst P@5"]
            )
            dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
            test_MAP = test_MRR = test_P1 = test_P5 = 0

            result_table_tp = PrettyTable(
                ["Epoch", "dev A P", "dev A R", "dev A F1", "dev I P", "dev I R", "dev I F1",
                 "tst A P", "tst A R", "tst A F1", "tst I P", "tst I R", "tst I F1"]
            )
            dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1 = 0, 0, 0, 0, 0, 0
            test_MAC_P, test_MAC_R, test_MAC_F1, test_MIC_P, test_MIC_R, test_MIC_F1 = 0, 0, 0, 0, 0, 0

            best_dev_performance = -1

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

            loss_qa_summary = tf.summary.scalar("loss_qa", self.loss_qa)
            loss_tp_summary = tf.summary.scalar("loss_tp", self.loss_tp)
            cost_summary = tf.summary.scalar("cost", self.cost)

            train_summary_op = tf.summary.merge([loss_qa_summary, loss_tp_summary, cost_summary])
            train_summary_dir = os.path.join(self.args.save_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # DEV for QA
            dev_loss = tf.placeholder(tf.float32)
            dev_map = tf.placeholder(tf.float32)
            dev_mrr = tf.placeholder(tf.float32)
            dev_loss_summary = tf.summary.scalar("dev_loss", dev_loss)
            dev_map_summary = tf.summary.scalar("dev_map", dev_map)
            dev_mrr_summary = tf.summary.scalar("dev_mrr", dev_mrr)

            # DEV for TP
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
                [dev_loss_summary, dev_map_summary, dev_mrr_summary,
                 dev_mac_p_summary, dev_mic_p_summary,
                 dev_mac_r_summary, dev_mic_r_summary,
                 dev_mac_f1_summary, dev_mic_f1_summary]
            )
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

                train_loss_qa = 0.0
                train_loss_tp = 0.0
                train_cost = 0.0

                for i in xrange(N):
                    cur_step, cur_loss_qa, cur_loss_tp, cur_cost = self.train_batch(
                        train_batches[i], train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss_qa += cur_loss_qa
                    train_loss_tp += cur_loss_tp
                    train_cost += cur_cost

                    if i % 10 == 0:
                        say("\r{}/{}".format(i, N))

                    if i == N-1:  # EVAL
                        if dev:
                            dev_MAP, dev_MRR, dev_P1, dev_P5, dev_hinge_loss, (
                                dev_LRAP, dev_LRL, dev_CE,
                                (dev_MAC_P, dev_MAC_R, dev_MAC_F1),
                                (dev_MIC_P, dev_MIC_R, dev_MIC_F1)
                            ) = self.evaluate(dev, sess)

                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_loss: dev_hinge_loss, dev_map: dev_MAP, dev_mrr: dev_MRR,
                                 dev_mac_f1: dev_MAC_F1, dev_mic_f1: dev_MIC_F1,
                                 dev_mac_p: dev_MAC_P, dev_mic_p: dev_MIC_P,
                                 dev_mac_r: dev_MAC_R, dev_mic_r: dev_MIC_R},
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            test_MAP, test_MRR, test_P1, test_P5, test_hinge_loss, (
                                test_LRAP, test_LRL, test_CE,
                                (test_MAC_P, test_MAC_R, test_MAC_F1),
                                (test_MIC_P, test_MIC_R, test_MIC_F1)
                            ) = self.evaluate(test, sess)

                        if self.args.performance == "f1_micro" and dev_MIC_F1 > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_MIC_F1
                            result_table_qa.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1,
                                 test_MAC_P, test_MAC_R, test_MAC_F1, test_MIC_P, test_MIC_R, test_MIC_F1]
                            )

                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "mrr" and dev_MRR > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_MRR
                            result_table_qa.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_MAC_P, dev_MAC_R, dev_MAC_F1, dev_MIC_P, dev_MIC_R, dev_MIC_F1,
                                 test_MAC_P, test_MAC_R, test_MAC_F1, test_MIC_P, test_MIC_R, test_MIC_F1]
                            )

                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        say("\r\n\nEpoch {}\tcost={:.3f}\tloss_qa={:.3f}\tloss_tp={:.3f}\tMRR={:.2f},{:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss_qa / (i+1),  # i.e. divided by N training batches
                            train_loss_tp / (i+1),  # i.e. divided by N training batches
                            dev_MRR,
                            best_dev_performance
                        ))
                        say("\n{}\n".format(result_table_qa))
                        say("\n{}\n".format(result_table_tp))

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


class LstmQATP(ModelQATP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}

    def _initialize_encoder_graph(self):

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

            if self.args.normalize:
                self.t_states_series = self.normalize_3d(self.t_states_series)

            if self.args.average:
                self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
            else:
                self.t_state = self.t_current_state[0][1]

        with tf.name_scope('bodies_output'):

            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                cell,
                self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            else:
                self.b_state = self.b_current_state[0][1]

        with tf.name_scope('outputs'):
            with tf.name_scope('encodings'):
                # batch * d
                h_final = (self.t_state + self.b_state) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s

    def average_without_padding(self, x, ids, eps=1e-8):
        # len*batch*1
        mask = tf.not_equal(ids, self.padding_id)
        mask = tf.expand_dims(mask, 2)
        mask = tf.cast(mask, tf.float32)
        # batch*d
        s = tf.reduce_sum(x*mask, axis=1) / (tf.reduce_sum(mask, axis=1)+eps)
        return s


class BiLstmQA(ModelQATP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}

    def _initialize_encoder_graph(self):

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

            def lstm_cell(state_size):
                _cell = tf.nn.rnn_cell.LSTMCell(
                    state_size, state_is_tuple=True, activation=get_activation_by_name(self.args.activation)
                )
                # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                return _cell

            forward_cell = lstm_cell(self.args.hidden_dim / 2 if self.args.concat == 1 else self.args.hidden_dim)
            backward_cell = lstm_cell(self.args.hidden_dim / 2 if self.args.concat == 1 else self.args.hidden_dim)

        with tf.name_scope('titles_output'):
            t_outputs, t_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=self.titles,
                dtype=tf.float32,
                sequence_length=self.SLT
            )

            forw_t_outputs, back_t_outputs = t_outputs
            forw_t_state, back_t_state = t_state

            if self.args.normalize:
                forw_t_outputs = self.normalize_3d(forw_t_outputs)
                back_t_outputs = self.normalize_3d(back_t_outputs)

            if self.args.average:
                forw_t_state = self.average_without_padding(forw_t_outputs, self.titles_words_ids_placeholder)
                back_t_state = self.average_without_padding(back_t_outputs, self.titles_words_ids_placeholder)
            else:
                forw_t_state = forw_t_state[1]  # (this is last output based on seq len)
                back_t_state = back_t_state[1]  # (same BUT in backwards => first output!)

            if self.args.concat:
                self.t_state_vec = tf.concat([forw_t_state, back_t_state], axis=1)
            else:
                self.t_state_vec = (forw_t_state + back_t_state) / 2.

        with tf.name_scope('bodies_output'):
            b_outputs, b_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_cell,
                cell_bw=backward_cell,
                inputs=self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )

            forw_b_outputs, back_b_outputs = b_outputs
            forw_b_state, back_b_state = b_state

            if self.args.normalize:
                forw_b_outputs = self.normalize_3d(forw_b_outputs)
                back_b_outputs = self.normalize_3d(back_b_outputs)

            if self.args.average:
                forw_b_state = self.average_without_padding(forw_b_outputs, self.bodies_words_ids_placeholder)
                back_b_state = self.average_without_padding(back_b_outputs, self.bodies_words_ids_placeholder)
            else:
                forw_b_state = forw_b_state[1]  # (this is last output based on seq len)
                back_b_state = back_b_state[1]  # (same BUT in backwards => first output!)

            if self.args.concat:
                self.b_state_vec = tf.concat([forw_b_state, back_b_state], axis=1)
            else:
                self.b_state_vec = (forw_b_state + back_b_state) / 2.

        with tf.name_scope('outputs'):
            with tf.name_scope('encodings'):
                # batch * d
                h_final = (self.t_state_vec + self.b_state_vec) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s

    def average_without_padding(self, x, ids, eps=1e-8):
        # len*batch*1
        mask = tf.not_equal(ids, self.padding_id)
        mask = tf.expand_dims(mask, 2)
        mask = tf.cast(mask, tf.float32)
        # batch*d
        s = tf.reduce_sum(x*mask, axis=1) / (tf.reduce_sum(mask, axis=1)+eps)
        return s


class CnnQA(ModelQATP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}

    def _initialize_encoder_graph(self):

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

                        nl_fun = get_activation_by_name(self.args.activation)
                        h_t = nl_fun(tf.nn.bias_add(conv_t, b), name="act-titles")

                        if self.args.average:
                            pooled_t = tf.reduce_mean(
                                h_t,
                                axis=1,
                                keep_dims=True
                            )
                        else:
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

                        nl_fun = get_activation_by_name(self.args.activation)
                        h_b = nl_fun(tf.nn.bias_add(conv_b, b), name="act-bodies")

                        if self.args.average:
                            pooled_b = tf.reduce_mean(
                                h_b,
                                axis=1,
                                keep_dims=True
                            )
                        else:
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
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)


class GruQA(ModelQATP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}

    def _initialize_encoder_graph(self):

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

        with tf.name_scope('GRU'):

            def gru_cell():
                _cell = tf.nn.rnn_cell.GRUCell(
                    self.args.hidden_dim, activation=get_activation_by_name(self.args.activation)
                )
                # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                return _cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [gru_cell() for _ in range(self.args.depth)]
            )

        with tf.name_scope('titles_output'):
            self.t_states_series, self.t_current_state = tf.nn.dynamic_rnn(
                cell,
                self.titles,
                dtype=tf.float32,
                sequence_length=self.SLT
            )

            if self.args.normalize:
                self.t_states_series = self.normalize_3d(self.t_states_series)

            if self.args.average:
                self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
            else:
                self.t_state = self.t_current_state[0]

        with tf.name_scope('bodies_output'):
            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                cell,
                self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            else:
                self.b_state = self.b_current_state[0]

        with tf.name_scope('outputs'):
            # batch * d
            h_final = (self.t_state + self.b_state) * 0.5
            h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
            self.h_final = self.normalize_2d(h_final)

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s

    def average_without_padding(self, x, ids, eps=1e-8):
        # len*batch*1
        mask = tf.not_equal(ids, self.padding_id)
        mask = tf.expand_dims(mask, 2)
        mask = tf.cast(mask, tf.float32)
        # batch*d
        s = tf.reduce_sum(x*mask, axis=1) / (tf.reduce_sum(mask, axis=1)+eps)
        return s
