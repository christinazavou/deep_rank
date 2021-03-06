import tensorflow as tf
import numpy as np
from qr.evaluation import Evaluation as QAEvaluation
from tags_prediction.evaluation import Evaluation as TPEvaluation
from nn import get_activation_by_name, init_w_b_vals
import gzip
import pickle
from prettytable import PrettyTable
from qr.myio import say
import os
import myio
from losses import qrdev_entropy_loss, qrdevloss0, qrdevloss0sum, qrdevloss1, qrdevloss2, qrdevloss2sum, qrloss0, \
    qrloss0sum, qrloss1, qrloss2, qrloss2sum
from losses import tpdev_entropy_loss, tpdev_hinge_loss, tpentropy_loss, tphinge_loss, tptrialloss


class ModelQRTP(object):

    def ready(self):
        self._initialize_placeholders_graph()
        self._initialize_encoder_graph()
        self._initialize_output_graph_qa()
        self._initialize_output_graph_tp()
        for param in tf.trainable_variables():
            self.params[param.name] = param
        self.params[self.embeddings.name] = self.embeddings  # in case it is not trainable
        self._initialize_cost_function()

    def _initialize_placeholders_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22
            self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='target_tags')

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            self.tuples = tf.placeholder(tf.int32, [None, None], name='tuples')
            self.target_scores = tf.placeholder(tf.float32, [None, None], name='target_scores')
            # self.tag_samples = tf.placeholder(tf.int32, [None, None], name='tag_samples')

    def _initialize_output_graph_qa(self):

        with tf.name_scope('scores'):
            self.scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)
            # triples_len x 22 x hidden_dim
            pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
            query_vecs = pairs_vecs[:, 0, :]

            if 'mlp_dim' in self.args and self.args.mlp_dim != 0:
                pos_scores, all_neg_scores = self.mlp(query_vecs, pairs_vecs[:, 1, :], pairs_vecs[:, 2:, :])
            else:
                # [num_of_tuples]
                pos_scores = tf.reduce_sum(query_vecs * pairs_vecs[:, 1, :], axis=1)
                # [num_of_tuples, candidate size - 1]
                all_neg_scores = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 2:, :], axis=2)

        with tf.name_scope('QaLoss'):

            if 'mlp_dim_qr' in self.args and self.args.mlp_dim_qr != 0:
                x_entropy_pos = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(pos_scores),
                    logits=pos_scores
                )
                x_entropy_neg = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(all_neg_scores),
                    logits=all_neg_scores
                )
                pos_weight = 1.  # allows one to trade off recall and precision
                self.loss_qr = (tf.reduce_mean(x_entropy_pos) * pos_weight + tf.reduce_mean(x_entropy_neg)) * 0.5
            else:
                if 'loss_qr' in self.args and self.args.loss_qr == 'loss2':
                    self.loss_qr = qrloss2(pos_scores, all_neg_scores)
                else:
                    self.loss_qr = qrloss0(pos_scores, all_neg_scores)

    # [tuples_num, hidden_dim],[tuples_num, hidden_dim], [tuples_num, 20, hidden_dim]
    def mlp(self, queries_vec, positives_vec, negatives_vec):

        left = np.random.random() < 0.5

        # [tuples_num, 20, hidden_dim]
        q_exp = tf.tile(tf.reshape(queries_vec, (-1, 1, self.args.hidden_dim)), (1, 20, 1))

        # [tuples_num, hidden_dim*2]
        if left:
            q_vec_p_vec = tf.reshape(tf.stack([queries_vec, positives_vec], axis=1), (-1, 2*self.args.hidden_dim))
        else:
            q_vec_p_vec = tf.reshape(tf.stack([positives_vec, queries_vec], axis=1), (-1, 2*self.args.hidden_dim))  # REV.

        # [tuples_num*20, hidden_dim*2]
        if left:
            q_vecs_n_vecs = tf.reshape(tf.concat([q_exp, negatives_vec], axis=2), (-1, 2*self.args.hidden_dim))
        else:
            q_vecs_n_vecs = tf.reshape(tf.concat([negatives_vec, q_exp], axis=2), (-1, 2*self.args.hidden_dim))  # REV.

        with tf.name_scope('MLP_QR'):

            def make_layer(in_dim, out_dim, name_w, name_b):
                w_val, b_val = init_w_b_vals([in_dim, out_dim], [out_dim], self.args.activation)
                w, b = tf.Variable(w_val, name=name_w), tf.Variable(b_val, name=name_b)
                return w, b

            def forward_layer(inp, w, b, with_activation=True):
                inp = tf.nn.dropout(inp, 1.-self.dropout_prob)
                outp = tf.add(tf.matmul(inp, w), b)
                if with_activation:
                    outp = tf.nn.relu(outp)
                return outp

            weights_h1, biases_h1 = make_layer(self.args.hidden_dim*2, self.args.mlp_dim, 'weights_h1', 'bias_h1')
            act_h_layer_p = forward_layer(q_vec_p_vec, weights_h1, biases_h1, True)
            act_h_layer_n = forward_layer(q_vecs_n_vecs, weights_h1, biases_h1, True)

            weights_o, biases_o = make_layer(self.args.mlp_dim, 1, 'weights_o', 'bias_o')
            output_p = forward_layer(act_h_layer_p, weights_o, biases_o, False)
            output_n = forward_layer(act_h_layer_n, weights_o, biases_o, False)

        return tf.squeeze(output_p), tf.reshape(output_n, (-1, 20))

    def _initialize_output_graph_tp(self):

        with tf.name_scope('outputs'):

            if self.args.mlp_dim_tp != 0:
                scope_name = 'MLP_TP'
            else:
                scope_name = 'SLP_TP'

            with tf.name_scope(scope_name):

                if self.args.mlp_dim_tp != 0:
                    w_h1, b_h1 = init_w_b_vals(
                        [self.args.hidden_dim, self.args.mlp_dim_tp], [self.args.mlp_dim_tp], self.args.activation)
                    weights_h1, biases_h1 = tf.Variable(w_h1, name='weights_h1'), tf.Variable(b_h1, name='bias_h1')
                    if self.args.depth > 1:
                        assert self.args.layer in ['lstm', 'gru']
                        layer_1 = tf.add(tf.matmul(self.mid_final, weights_h1), biases_h1)
                    else:
                        layer_1 = tf.add(tf.matmul(self.h_final, weights_h1), biases_h1)
                    act_layer_1 = tf.nn.relu(layer_1)  # to reduce training time

                    w_o, b_o = init_w_b_vals([self.args.mlp_dim_tp, self.output_dim], [self.output_dim], self.args.activation)
                    weights_o, biases_o = tf.Variable(w_o, name='weights_o'), tf.Variable(b_o, name='bias_o')
                    output = tf.matmul(act_layer_1, weights_o) + biases_o

                else:
                    w_o, b_o = init_w_b_vals([self.args.hidden_dim, self.output_dim], [self.output_dim], self.args.activation)
                    weights_o, biases_o = tf.Variable(w_o, name='weights_o'), tf.Variable(b_o, name='bias_o')
                    if self.args.depth > 1:
                        assert self.args.layer in ['lstm', 'gru']
                        output = tf.matmul(self.mid_final, weights_o) + biases_o
                    else:
                        output = tf.matmul(self.h_final, weights_o) + biases_o

                self.b_o = biases_o  # for api compatibility

                self.act_output = tf.nn.sigmoid(output)

        with tf.name_scope('TpLoss'):
            if 'entropy_tp' not in self.args or self.args.entropy_tp == 1:
                self.loss_tp = tpentropy_loss(self.args, self.target, self.act_output)
            else:

                if 'loss_tp' not in self.args or self.args.loss_tp == 'loss2':
                    self.loss_tp = tphinge_loss(self.target, self.act_output, self.tuples)
                else:
                    self.loss_tp = tphinge_loss(self.target, self.act_output, self.tuples, take_max=True)

                # self.trial_loss = tptrialloss(self.target, self.act_output, self.tag_samples)

    def _initialize_cost_function(self):
        with tf.name_scope('cost'):
            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in set(tf.trainable_variables() + [self.embeddings]):  # in case not trainable emb
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg
            self.loss_qr *= self.args.qr_weight  # to visualize in Tensorboard
            self.loss_tp *= self.args.tp_weight  # to visualize in Tensorboard
            self.cost = self.loss_qr + self.loss_tp + self.l2_reg

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

    def get_pnorm_stat(self, session):
        dict_norms = {}
        for param_name, param in self.params.iteritems():
            l2 = session.run(tf.norm(param))
            dict_norms[param_name] = round(l2, 3)
        return dict_norms

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
        scores, outputs = sess.run(
            [self.scores, self.act_output],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return scores, outputs

    def evaluate(self, data, sess):
        res = []
        all_labels = []
        all_scores = []

        outputs, targets = [], []
        tuples = []

        sample = 0
        for idts, idbs, id_labels, tags_b, tuples_b in data:
            sample += 1
            cur_scores, cur_out = self.eval_batch(idts, idbs, sess)
            outputs.append(cur_out)
            targets.append(tags_b)
            tuples.append(tuples_b)

            assert len(id_labels) == len(cur_scores)  # equal to 20

            all_labels.append(id_labels)
            all_scores.append(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = QAEvaluation(res)
        MAP = e.MAP()
        MRR = e.MRR()
        P1 = e.Precision(1)
        P5 = e.Precision(5)

        if 'entropy_qr' in self.args and self.args.entropy_qr != 0:
            loss_qr = qrdev_entropy_loss(all_labels, all_scores)  # todo : encounters nan
        else:
            loss_qr = qrdevloss0(all_labels, all_scores)
        loss_qr *= self.args.qr_weight

        outputs = np.vstack(outputs)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object
        tuples = np.vstack(tuples)

        if 'entropy_tp' not in self.args or self.args.entropy_tp == 1:
            loss_tp = tpdev_entropy_loss(self.args, targets, outputs)
        else:
            loss_tp = tpdev_hinge_loss(targets, outputs, tuples)
        loss_tp *= self.args.tp_weight

        """------------------------------------------remove ill evaluation-------------------------------------------"""
        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, targets = outputs[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""

        ev = TPEvaluation(outputs, None, targets)
        results = [ev.Precision(5), ev.Precision(10), ev.Recall(5), ev.Recall(10), ev.MeanAveragePrecision()]

        return MAP, MRR, P1, P5, loss_qr, loss_tp, tuple(results)

    def train_batch(self, batch, train_op, global_step, sess):
        titles, bodies, pairs, tags, tuples = batch
        target_scores = np.zeros((len(pairs), 21))
        target_scores[:, 0] = 1.
        _, _step, _loss_qr, _loss_tp, _cost = sess.run(
            [train_op, global_step, self.loss_qr, self.loss_tp, self.cost],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.target: tags,
                self.dropout_prob: np.float32(self.args.dropout),
                self.tuples: tuples,
                self.target_scores: target_scores,
                # self.tag_samples: tag_samples
            }
        )
        return _step, _loss_qr, _loss_tp, _cost

    def train_model(self, ids_corpus_tags, train, dev=None, test=None):
        with tf.Session() as sess:

            result_table_qr = PrettyTable(
                ["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5", "tst MAP", "tst MRR", "tst P@1", "tst P@5"]
            )
            dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
            test_MAP = test_MRR = test_P1 = test_P5 = 0

            result_table_tp = PrettyTable(
                ["Epoch", "dev P@5", "dev P@10", "dev R@5", "dev R@10", "dev MAP",
                 "tst P@5", "tst P@10", "tst R@5", "tst R@10", "tst MAP"]
            )

            dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_map = 0, 0, 0, 0, 0
            test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_map = 0, 0, 0, 0, 0

            best_qr_dev_performance = -1
            best_tp_dev_performance = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
            # todo: two different optimizers, l2_reg, learning_rates
            train_op = optimizer.minimize(self.cost, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            emb = sess.run(self.embeddings)
            print '\nemb {}\n'.format(emb[10][0:10])

            if self.init_assign_ops != {}:
                print 'assigning trained values ...\n'
                sess.run(self.init_assign_ops)
                emb = sess.run(self.embeddings)
                print '\nemb {}\n'.format(emb[10][0:10])
                self.init_assign_ops = {}

            if self.args.save_dir != "":
                print("Writing to {}\n".format(self.args.save_dir))

            # TRAIN LOSS
            train_loss_qr_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "QR"),
            )
            train_loss_tp_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "TP"),
            )
            train_cost_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train"), sess.graph
            )

            # VARIABLE NORM
            p_norm_summaries = {}
            p_norm_placeholders = {}
            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                p_norm_placeholders[param_name] = tf.placeholder(tf.float32)
                p_norm_summaries[param_name] = tf.summary.scalar(param_name, p_norm_placeholders[param_name])
            p_norm_summary_op = tf.summary.merge(p_norm_summaries.values())
            p_norm_summary_dir = os.path.join(self.args.save_dir, "summaries", "p_norm")
            p_norm_summary_writer = tf.summary.FileWriter(p_norm_summary_dir, )

            # DEV LOSS
            dev_loss_qr_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "QR"),
            )
            dev_loss_tp_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP"),
            )

            # DEV evaluation for QR
            dev_eval_qr_writer1 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "QR", "MAP"),
            )
            dev_eval_qr_writer2 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "QR", "MRR"),
            )
            dev_eval_qr_writer3 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "QR", "Pat1"),
            )
            dev_eval_qr_writer4 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "QR", "Pat5"),
            )

            # DEV for TP
            dev_eval_tp_writer1 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP", "Rat5"),
            )
            dev_eval_tp_writer2 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP", "Rat10"),
            )
            dev_eval_tp_writer3 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP", "Pat5"),
            )
            dev_eval_tp_writer4 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP", "Pat10"),
            )
            dev_eval_tp_writer5 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "TP", "MAP"),
            )

            loss = tf.placeholder(tf.float32)
            loss_summary = tf.summary.scalar("loss", loss)
            dev_eval = tf.placeholder(tf.float32)
            dev_qr_summary = tf.summary.scalar("QR_evaluation", dev_eval)
            dev_tp_summary = tf.summary.scalar("TP_evaluation", dev_eval)
            cost = tf.placeholder(tf.float32)
            cost_summary = tf.summary.scalar("cost", cost)

            if self.args.save_dir != "":
                checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

            train_batches = list(myio.create_batches(
                ids_corpus_tags, train, self.args.batch_size, self.padding_id, N_neq=self.args.n_neg, samples_file=self.args.samples_file
            ))
            N = len(train_batches)

            unchanged = 0
            max_epoch = self.args.max_epoch
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > 15:
                    break

                train_batches = myio.create_batches(
                    ids_corpus_tags, train, self.args.batch_size, self.padding_id, N_neq=self.args.n_neg
                )

                train_loss_qr = 0.0
                train_loss_tp = 0.0
                train_cost = 0.0

                i = -1
                for batch in train_batches:
                    i += 1
                    cur_step, cur_loss_qr, cur_loss_tp, cur_cost = self.train_batch(
                        batch, train_op, global_step, sess
                    )

                    summary = sess.run(loss_summary, {loss: cur_loss_qr})
                    train_loss_qr_writer.add_summary(summary, cur_step)
                    train_loss_qr_writer.flush()
                    summary = sess.run(loss_summary, {loss: cur_loss_tp})
                    train_loss_tp_writer.add_summary(summary, cur_step)
                    train_loss_tp_writer.flush()
                    summary = sess.run(cost_summary, {cost: cur_cost})
                    train_cost_writer.add_summary(summary, cur_step)
                    train_cost_writer.flush()

                    train_loss_qr += cur_loss_qr
                    train_loss_tp += cur_loss_tp
                    train_cost += cur_cost

                    if i % 10 == 0:
                        say("\r{}/{}".format(i, N))
                    if self.args.testing:
                        print 'labels in batch: ', np.sum(np.sum(batch[3], 0) > 0)

                    if i == N-1 or (i % 10 == 0 and self.args.testing):  # EVAL
                        dev_tp_loss = 0
                        dev_qr_loss = 0

                        if dev:
                            dev_MAP, dev_MRR, dev_P1, dev_P5, dev_qr_loss, dev_tp_loss, (
                                dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_map
                            ) = self.evaluate(dev, sess)

                            summary = sess.run(loss_summary, {loss: dev_qr_loss})
                            dev_loss_qr_writer.add_summary(summary, cur_step)
                            dev_loss_qr_writer.flush()
                            summary = sess.run(loss_summary, {loss: dev_tp_loss})
                            dev_loss_tp_writer.add_summary(summary, cur_step)
                            dev_loss_tp_writer.flush()

                            summary = sess.run(dev_qr_summary, {dev_eval: dev_MAP})
                            dev_eval_qr_writer1.add_summary(summary, cur_step)
                            dev_eval_qr_writer1.flush()
                            summary = sess.run(dev_qr_summary, {dev_eval: dev_MRR})
                            dev_eval_qr_writer2.add_summary(summary, cur_step)
                            dev_eval_qr_writer2.flush()
                            summary = sess.run(dev_qr_summary, {dev_eval: dev_P1})
                            dev_eval_qr_writer3.add_summary(summary, cur_step)
                            dev_eval_qr_writer3.flush()
                            summary = sess.run(dev_qr_summary, {dev_eval: dev_P5})
                            dev_eval_qr_writer4.add_summary(summary, cur_step)
                            dev_eval_qr_writer4.flush()

                            summary = sess.run(dev_tp_summary, {dev_eval: dev_RAT5})
                            dev_eval_tp_writer1.add_summary(summary, cur_step)
                            dev_eval_tp_writer1.flush()
                            summary = sess.run(dev_tp_summary, {dev_eval: dev_RAT10})
                            dev_eval_tp_writer2.add_summary(summary, cur_step)
                            dev_eval_tp_writer2.flush()
                            summary = sess.run(dev_tp_summary, {dev_eval: dev_PAT5})
                            dev_eval_tp_writer3.add_summary(summary, cur_step)
                            dev_eval_tp_writer3.flush()
                            summary = sess.run(dev_tp_summary, {dev_eval: dev_PAT10})
                            dev_eval_tp_writer4.add_summary(summary, cur_step)
                            dev_eval_tp_writer4.flush()
                            summary = sess.run(dev_tp_summary, {dev_eval: dev_map})
                            dev_eval_tp_writer5.add_summary(summary, cur_step)
                            dev_eval_tp_writer5.flush()

                            feed_dict = {}
                            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                                feed_dict[p_norm_placeholders[param_name]] = param_norm
                            _p_norm_sum = sess.run(p_norm_summary_op, feed_dict)
                            p_norm_summary_writer.add_summary(_p_norm_sum, cur_step)

                        if test:
                            test_MAP, test_MRR, test_P1, test_P5, test_qr_loss, test_tp_loss, (
                                test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_map
                            ) = self.evaluate(test, sess)

                        if self.args.performance == "dev_map_qr" and dev_MAP > best_qr_dev_performance:
                            unchanged = 0
                            best_qr_dev_performance = dev_MAP
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_map,
                                 test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_map]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "dev_map_tp" and dev_map > best_tp_dev_performance:
                            unchanged = 0
                            best_tp_dev_performance = dev_map
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_map,
                                 test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_map]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "" and (dev_MAP > best_qr_dev_performance or dev_map > best_tp_dev_performance):
                            unchanged = 0
                            best_tp_dev_performance = dev_map if dev_map > best_tp_dev_performance else best_tp_dev_performance
                            best_qr_dev_performance = dev_MAP if dev_MAP > best_qr_dev_performance else best_qr_dev_performance
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_map,
                                 test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_map]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        say(
                            "\r\n\nEpoch {}:\tcost={:.3f}, loss_qr={:.3f}, loss_tp={:.3f}, "
                            "devMRR={:.3f}, DevLossQR={:.3f}, DevLossTP={:.3f}\n".format(
                                epoch,
                                train_cost / (i+1),  # i.e. divided by N training batches
                                train_loss_qr / (i+1),  # i.e. divided by N training batches
                                train_loss_tp / (i+1),  # i.e. divided by N training batches
                                dev_MRR,
                                dev_qr_loss,
                                dev_tp_loss,
                            )
                        )
                        say("P@5 {} R@10 {}\n".format(dev_PAT5, dev_RAT10))
                        say("\n{}\n".format(result_table_qr))
                        say("\n{}\n".format(result_table_tp))
                        say("\tp_norm: {}\n".format(
                            self.get_pnorm_stat(sess)
                        ))

    def save(self, sess, path, step):
        # NOTE: Optimizer is not saved!!! So if more train..optimizer starts again
        path = "{}_{}".format(path, step, ".pkl.gz")
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
            assert self.args.hidden_dim == data["args"].hidden_dim, 'you are trying to load model with {} hid-dim, '\
                'while your model has {} hid dim'.format(data["args"].hidden_dim, self.args.hidden_dim)
            params_values = data['params_values']
            for param_name, param_value in params_values.iteritems():
                if (self.args.load_only_embeddings == 1) and (param_name != self.embeddings.name):
                    print 'skip {} reason: load only embeddings'.format(param_name)
                    continue
                if param_name in self.params:
                    print param_name, ' is in my dict'
                    variable = self.params[param_name]
                    assign_ops[param_name] = tf.assign(variable, param_value)
                else:
                    print param_name, ' is not in my dict'
        return assign_ops

    def load_trained_vars(self, path):
        print("Loading model checkpoint from {}\n".format(path))
        assert self.args is not None and self.params != {}
        assign_ops = {}
        with gzip.open(path) as fin:
            data = pickle.load(fin)
            print("WARNING: hid dim ({}) != pre trained model hid dim ({}). Use {} instead.\n".format(
                self.args.hidden_dim, data["args"].hidden_dim, data["args"].hidden_dim
            ))
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
            print 'assigning values in ', param_name
            sess.run(param_assign_op)
        self.init_assign_ops = {}  # to avoid reassigning embeddings if train

    def num_parameters(self):  # this is not just the trainable params if embeddings are not trainable p.x.
        total_parameters = 0
        for param_name, param in self.params.iteritems():
            # shape is an array of tf.Dimension
            shape = param.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


"""--------------------------------------------------RNN Models------------------------------------------------------"""


class RNNQRTP(ModelQRTP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        if embedding_layer.init_embeddings is not None:
            assign_op = tf.assign(self.embeddings, embedding_layer.init_embeddings)
            self.init_assign_ops = {self.embeddings: assign_op}
        else:
            self.init_assign_ops = {}

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

    def maximum_without_padding(self, x, ids):

        def tf_repeat(tensor, repeats):
            expanded_tensor = tf.expand_dims(tensor, -1)
            multiples = [1] + repeats
            tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
            repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
            return repeated_tensor

        # len*batch*1
        mask = tf.not_equal(ids, self.padding_id)
        condition = tf.reshape(tf_repeat(mask, [1, tf.shape(x)[2]]), tf.shape(x))

        smallest = tf.ones_like(x)*(-100000)

        # batch*d
        s = tf.where(condition, x, smallest)
        m = tf.reduce_max(s, 1)
        return m


"""--------------------------------------------------LSTM Model------------------------------------------------------"""


class LstmQRTP(RNNQRTP):

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
                    self.args.hidden_dim,
                    state_is_tuple=True,
                    activation=get_activation_by_name(self.args.activation)
                )
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
            # i.e. [batch_size, max_time, cell.output_size]
            if self.args.depth > 1:
                assert self.args.depth == 2
                print 'depth > 1 => will use last pooling'
                self.mid_t_state = self.t_current_state[0][1]
                self.t_state = self.t_current_state[1][1]
            else:
                if self.args.normalize:
                    self.t_states_series = self.normalize_3d(self.t_states_series)

                if self.args.average == 1:
                    self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
                elif self.args.average == 0:
                    # self.t_state=self.t_states_series[:, -1, :]=self.t_current_state[-1][1]=self.t_current_state[0][1]
                    # in case sequence_length parameter is used in RNN, the last state is not self.t_states_series[:,-1,:]
                    # but is self.t_states_series[:, self.SLT[x], :] and it is stored correctly in
                    # self.t_current_state[0][1] so its better and safer to use this.
                    self.t_state = self.t_current_state[0][1]
                else:
                    self.t_state = self.maximum_without_padding(self.t_states_series, self.titles_words_ids_placeholder)

        with tf.name_scope('bodies_output'):

            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                cell,
                self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )

            if self.args.depth > 1:
                assert self.args.depth == 2
                print 'depth > 1 => will use last pooling'
                self.mid_b_state = self.b_current_state[0][1]
                self.b_state = self.b_current_state[1][1]
            else:

                if self.args.normalize:
                    self.b_states_series = self.normalize_3d(self.b_states_series)

                if self.args.average == 1:
                    self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
                elif self.args.average == 0:
                    self.b_state = self.b_current_state[0][1]
                else:
                    self.b_state = self.maximum_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)

        with tf.name_scope('outputs'):
            with tf.name_scope('encodings'):
                if self.args.depth > 1:
                    mid_final = (self.mid_t_state + self.mid_b_state) * 0.5
                    mid_final = tf.nn.dropout(mid_final, 1. - self.dropout_prob)
                    self.mid_final = self.normalize_2d(mid_final)
                # batch * d
                h_final = (self.t_state + self.b_state) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)


"""----------------------------------------------------GRU Model-----------------------------------------------------"""


class GruQRTP(RNNQRTP):

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
            if self.args.depth > 1:
                assert self.args.depth == 2
                print 'depth > 1 => will use last pooling'
                self.mid_t_state = self.t_current_state[0]
                self.t_state = self.t_current_state[1]
            else:

                if self.args.normalize:
                    self.t_states_series = self.normalize_3d(self.t_states_series)

                if self.args.average == 1:
                    self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
                elif self.args.average == 0:
                    self.t_state = self.t_current_state[0]
                else:
                    self.t_state = self.maximum_without_padding(self.t_states_series, self.titles_words_ids_placeholder)

        with tf.name_scope('bodies_output'):
            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                cell,
                self.bodies,
                dtype=tf.float32,
                sequence_length=self.SLB
            )
            if self.args.depth > 1:
                assert self.args.depth == 2
                print 'depth > 1 => will use last pooling'
                self.mid_b_state = self.b_current_state[0]
                self.b_state = self.b_current_state[1]
            else:

                if self.args.normalize:
                    self.b_states_series = self.normalize_3d(self.b_states_series)

                if self.args.average == 1:
                    self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
                elif self.args.average == 0:
                    self.b_state = self.b_current_state[0]
                else:
                    self.b_state = self.maximum_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)

        with tf.name_scope('outputs'):
            if self.args.depth > 1:
                mid_final = (self.mid_t_state + self.mid_b_state) * 0.5
                mid_final = tf.nn.dropout(mid_final, 1. - self.dropout_prob)
                self.mid_final = self.normalize_2d(mid_final)
            # batch * d
            h_final = (self.t_state + self.b_state) * 0.5
            h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
            self.h_final = self.normalize_2d(h_final)


"""--------------------------------------------------BiRNN Models----------------------------------------------------"""


class BiRNNQRTP(RNNQRTP):

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

        if self.args.layer.lower() == 'bigru':
            with tf.name_scope('GRU'):

                def gru_cell(state_size):
                    _cell = tf.nn.rnn_cell.GRUCell(
                        state_size, activation=get_activation_by_name(self.args.activation)
                    )
                    # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                    return _cell

                forward_cell = gru_cell(self.args.hidden_dim / 2 if self.args.concat == 1 else self.args.hidden_dim)
                backward_cell = gru_cell(self.args.hidden_dim / 2 if self.args.concat == 1 else self.args.hidden_dim)
        else:
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

            if self.args.average == 1:
                forw_t_state = self.average_without_padding(forw_t_outputs, self.titles_words_ids_placeholder)
                back_t_state = self.average_without_padding(back_t_outputs, self.titles_words_ids_placeholder)
            elif self.args.average == 0:
                if self.args.layer.lower() == 'bigru':
                    forw_t_state = forw_t_state  # (this is last output based on seq len)
                    back_t_state = back_t_state  # (same BUT in backwards => first output!)
                else:
                    forw_t_state = forw_t_state[1]  # (this is last output based on seq len)
                    back_t_state = back_t_state[1]  # (same BUT in backwards => first output!)
            else:
                forw_t_state = self.maximum_without_padding(forw_t_outputs, self.titles_words_ids_placeholder)
                back_t_state = self.maximum_without_padding(back_t_outputs, self.titles_words_ids_placeholder)

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

            if self.args.average == 1:
                forw_b_state = self.average_without_padding(forw_b_outputs, self.bodies_words_ids_placeholder)
                back_b_state = self.average_without_padding(back_b_outputs, self.bodies_words_ids_placeholder)
            elif self.args.average == 0:
                if self.args.layer.lower() == 'bigru':
                    forw_b_state = forw_b_state  # (this is last output based on seq len)
                    back_b_state = back_b_state  # (same BUT in backwards => first output!)
                else:
                    forw_b_state = forw_b_state[1]  # (this is last output based on seq len)
                    back_b_state = back_b_state[1]  # (same BUT in backwards => first output!)
            else:
                forw_b_state = self.maximum_without_padding(forw_b_outputs, self.bodies_words_ids_placeholder)
                back_b_state = self.maximum_without_padding(back_b_outputs, self.bodies_words_ids_placeholder)

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


"""--------------------------------------------------CNN Model----------------------------------------------------"""


class CnnQRTP(ModelQRTP):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        if embedding_layer.init_embeddings is not None:
            assign_op = tf.assign(self.embeddings, embedding_layer.init_embeddings)
            self.init_assign_ops = {self.embeddings: assign_op}
        else:
            self.init_assign_ops = {}

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

                    w_vals, b_vals = init_w_b_vals(filter_shape, [self.args.hidden_dim], self.args.activation)
                    W = tf.Variable(w_vals, name="conv-W")
                    b = tf.Variable(b_vals, name="conv-b")

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
