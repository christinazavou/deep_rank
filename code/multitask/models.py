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

    def _initialize_output_graph_qa(self):

        with tf.name_scope('scores'):
            self.scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)
            # triples_len x 22 x hidden_dim
            pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
            query_vecs = pairs_vecs[:, 0, :]
            pos_scores = tf.reduce_sum(query_vecs * pairs_vecs[:, 1, :], axis=1)
            all_neg_scores = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 2:, :], axis=2)
            neg_scores = tf.reduce_max(all_neg_scores, axis=1)

        with tf.name_scope('QaLoss'):
            if 'entropy_qr' not in self.args or self.args.entropy_qr == 0:

                diff = neg_scores - pos_scores + 1.0
                # tf.cast((diff > 0), tf.float32) * diff is replacing in matrix diff the values <= 0 with zero
                if 'loss_qr' in self.args and self.args.loss_qr == 'max':
                    self.loss_qr = tf.reduce_max(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')
                elif 'loss_qr' in self.args and self.args.loss_qr == 'sum':
                    self.loss_qr = tf.reduce_sum(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')
                else:
                    self.loss_qr = tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')

                """-----------------------------modified version:-------------------------------"""
                # diff = all_neg_scores - tf.reshape(pos_scores, [-1, 1]) + 1.0
                # if 'loss' in self.args and self.args.loss == 'max':
                #     self.loss = tf.reduce_max(tf.reduce_sum(diff, 1), name='hinge_loss')
                # elif 'loss' in self.args and self.args.loss == 'sum':
                #     self.loss = tf.reduce_sum(tf.reduce_sum(diff, 1), name='hinge_loss')
                # else:
                #     self.loss = tf.reduce_mean(tf.reduce_sum(diff, 1), name='hinge_loss')

            else:

                raise Exception("dont use entropy")
                outputs = tf.concat([tf.reshape(pos_scores, [-1, 1]), all_neg_scores], 1)
                targets = tf.concat(
                    [tf.reshape(tf.ones_like(pos_scores), [-1, 1]), tf.zeros_like(all_neg_scores, tf.float32)], 1)
                # outputs lie in (0,1)
                x_entropy = targets * (-tf.log(outputs)) + (1.0 - targets) * (-tf.log(1.0 - outputs))
                if 'loss_qr' in self.args and self.args.loss_qr == "sum":
                    self.loss_qr = tf.reduce_sum(tf.reduce_sum(x_entropy, axis=1), name='x_entropy')
                elif 'loss_qr' in self.args and self.args.loss_qr == "max":
                    self.loss_qr = tf.reduce_max(tf.reduce_sum(x_entropy, axis=1), name='x_entropy')
                else:
                    self.loss_qr = tf.reduce_mean(tf.reduce_sum(x_entropy, axis=1), name='x_entropy')

            self.loss_qr *= self.args.qr_mul if 'qr_mul' in self.args else 1.  # version compatibility

    def _initialize_output_graph_tp(self):

        with tf.name_scope('outputs'):

            with tf.name_scope("MLP"):

                w_vals, b_vals = init_w_b_vals(
                    [self.args.hidden_dim, self.output_dim], [self.output_dim], self.args.activation
                )

                self.w_o = tf.Variable(w_vals, name='weights_out')
                self.b_o = tf.Variable(b_vals, name='bias_out')

            output = tf.matmul(self.h_final, self.w_o) + self.b_o
            self.act_output = tf.nn.sigmoid(output)

            # for evaluation
            self.prediction = tf.where(
                self.act_output > self.args.threshold, tf.ones_like(self.act_output), tf.zeros_like(self.act_output)
            )

        with tf.name_scope('TpLoss'):
            if 'entropy_tp' not in self.args or self.args.entropy_tp == 1:
                self.loss_tp = self.entropy_loss(output)
            else:
                self.loss_tp = self.hinge_loss()

            self.loss_tp *= self.args.tp_mul if 'tp_mul' in self.args else 1.  # version compatibility

    def entropy_loss(self, output):
        # x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=output)

        # tuplex_entropies = tf.nn.embedding_lookup(x_entropy, self.pairs_ids_placeholder, name='tuplex_entropies')
        # tuplex_entropies = tf.reduce_sum(tuplex_entropies, 2)
        # self.loss_entropy = tf.reduce_mean(tf.reduce_sum(tuplex_entropies, 1), name='tuplex_entropy')

        w = 1. if 'weight' not in self.args else self.args.weight
        weighted_entropy = self.target*(-tf.log(self.act_output))*w + (1.0-self.target)*(-tf.log(1.0-self.act_output))

        if 'loss_tp' in self.args and self.args.loss_tp == "sum":
            return tf.reduce_sum(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')
        elif 'loss_tp' in self.args and self.args.loss_tp == "max":
            return tf.reduce_max(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')
        else:
            return tf.reduce_mean(tf.reduce_sum(weighted_entropy, axis=1), name='x_entropy')

    def hinge_loss(self):
        raise Exception()
        min_pos = tf.reduce_min(self.target * self.act_output, 1)
        max_neg = tf.reduce_max(tf.cast(tf.equal(self.target, 0), tf.float32) * self.act_output, 1)
        diff = max_neg - min_pos + 1.0
        # tf.cast((diff > 0), tf.float32) * diff is replacing in matrix diff the values <= 0 with zero
        if 'loss_tp' in self.args and self.args.loss_tp == "sum":
            return tf.reduce_sum(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')
        elif 'loss_tp' in self.args and self.args.loss_tp == "max":
            return tf.reduce_max(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')
        else:
            return tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff, name='hinge_loss')

    def _initialize_cost_function(self):
        with tf.name_scope('cost'):
            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in set(tf.trainable_variables() + [self.embeddings]):  # in case not trainable emb
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg
            self.cost = self.args.qr_weight*self.loss_qr + self.args.tp_weight*self.loss_tp + self.l2_reg

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
        scores, outputs, predictions = sess.run(
            [self.scores, self.act_output, self.prediction],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return scores, outputs, predictions

    # todo: if binaryxentropy/hingeloss add...
    def evaluate(self, data, sess):
        res = []
        per_batch_losses = []
        per_query_losses = []

        outputs, predictions, targets = [], [], []

        sample = 0
        for idts, idbs, id_labels, tags_b in data:
            sample += 1
            cur_scores, cur_out, cur_pred = self.eval_batch(idts, idbs, sess)

            outputs.append(cur_out)
            predictions.append(cur_pred)
            targets.append(tags_b)

            mml = self.max_margin_loss(id_labels, cur_scores)
            if mml is not None:
                per_query_losses.append(mml)
            if (sample % self.args.batch_size == 0) or (sample == len(data) - 1):
                if 'loss_qr' in self.args and self.args.loss_qr == "sum":
                    hinge_loss = sum(per_query_losses)
                elif 'loss_qr' in self.args and self.args.loss_qr == "max":
                    hinge_loss = max(per_query_losses)
                else:
                    hinge_loss = sum(per_query_losses) / float(len(per_query_losses))
                per_batch_losses.append(hinge_loss)
                per_query_losses = []

            assert len(id_labels) == len(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)
        qr_loss = sum(per_batch_losses) / float(len(per_batch_losses))

        e = QAEvaluation(res)
        MAP = e.MAP()
        MRR = e.MRR()
        P1 = e.Precision(1)
        P5 = e.Precision(5)

        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object

        # outputs are passed through sigmoid, thus they lie in (0,1)
        x_entropy = targets * (-np.log(outputs)) + (1.0 - targets) * (-np.log(1.0 - outputs))
        if 'loss_tp' in self.args and self.args.loss_tp == "sum":
            tp_loss = np.sum(np.sum(x_entropy, 1))
        elif 'loss_tp' in self.args and self.args.loss_tp == "max":
            tp_loss = np.max(np.sum(x_entropy, 1))
        else:
            tp_loss = np.mean(np.sum(x_entropy, 1))

        """------------------------------------------remove ill evaluation-------------------------------------------"""
        # eval_labels = []
        # for label in range(targets.shape[1]):
        #     if (targets[:, label] == np.ones(targets.shape[0])).any():
        #         eval_labels.append(label)
        # print '\n{} labels out of {} will be evaluated (zero-sampled-labels removed).'.format(len(eval_labels), targets.shape[1])
        # outputs, predictions, targets = outputs[:, eval_labels], predictions[:, eval_labels], targets[:, eval_labels]

        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, predictions, targets = outputs[eval_samples, :], predictions[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""

        ev = TPEvaluation(outputs, predictions, targets)
        results = [ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10),
                   ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10)]
        return MAP, MRR, P1, P5, qr_loss, tp_loss, tuple(results)

    def train_batch(self, batch, train_op, global_step, sess):
        titles, bodies, pairs, tags = batch
        _, _step, _loss_qr, _loss_tp, _cost = sess.run(
            [train_op, global_step, self.loss_qr, self.loss_tp, self.cost],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.target: tags,
                self.dropout_prob: np.float32(self.args.dropout),
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
                ["Epoch", "dev P@1", "dev P@3", "dev P@5", "dev P@10", "dev R@1", "dev R@3", "dev R@5", "dev R@10",
                 "tst P@1", "tst P@3", "tst P@5", "tst P@10", "tst R@1", "tst R@3", "tst R@5", "tst R@10"]
            )

            dev_PAT1, dev_PAT3, dev_PAT5, dev_PAT10, dev_RAT1, dev_RAT3, dev_RAT5, dev_RAT10 = 0, 0, 0, 0, 0, 0, 0, 0
            test_PAT1, test_PAT3, test_PAT5, test_PAT10, test_RAT1, test_RAT3, test_RAT5, test_RAT10 = 0, 0, 0, 0, 0, 0, 0, 0

            best_dev_performance = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.args.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
            elif self.args.optimizer == "adagrad":
                optimizer = tf.train.AdagradOptimizer(self.args.learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.args.learning_rate)
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

            unchanged = 0
            max_epoch = self.args.max_epoch
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > 15:
                    break

                train_batches = myio.create_batches(ids_corpus_tags, train, self.args.batch_size, self.padding_id)

                N = len(train_batches)

                train_loss_qr = 0.0
                train_loss_tp = 0.0
                train_cost = 0.0

                for i in xrange(N):
                    cur_step, cur_loss_qr, cur_loss_tp, cur_cost = self.train_batch(
                        train_batches[i], train_op, global_step, sess
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
                        print 'labels in batch: ', np.sum(np.sum(train_batches[i][3], 0) > 0)

                    if i == N-1:  # EVAL
                        dev_tp_loss = 0
                        dev_qr_loss = 0

                        if dev:
                            dev_MAP, dev_MRR, dev_P1, dev_P5, dev_qr_loss, dev_tp_loss, (
                                dev_PAT1, dev_PAT3, dev_PAT5, dev_PAT10, dev_RAT1, dev_RAT3, dev_RAT5, dev_RAT10
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

                            feed_dict = {}
                            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                                feed_dict[p_norm_placeholders[param_name]] = param_norm
                            _p_norm_sum = sess.run(p_norm_summary_op, feed_dict)
                            p_norm_summary_writer.add_summary(_p_norm_sum, cur_step)

                        if test:
                            test_MAP, test_MRR, test_P1, test_P5, test_qr_loss, test_tp_loss, (
                                test_PAT1, test_PAT3, test_PAT5, test_PAT10, test_RAT1, test_RAT3, test_RAT5, test_RAT10
                            ) = self.evaluate(test, sess)

                        if self.args.performance == "dev_mrr" and dev_MRR > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_MRR
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT1, dev_PAT3, dev_PAT5, dev_PAT10, dev_RAT1, dev_RAT3, dev_RAT5, dev_RAT10,
                                 test_PAT1, test_PAT3, test_PAT5, test_PAT10, test_RAT1, test_RAT3, test_RAT5, test_RAT10]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "P@5" and dev_PAT5 > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_PAT5
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT1, dev_PAT3, dev_PAT5, dev_PAT10, dev_RAT1, dev_RAT3, dev_RAT5, dev_RAT10,
                                 test_PAT1, test_PAT3, test_PAT5, test_PAT10, test_RAT1, test_RAT3, test_RAT5, test_RAT10]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "R@10" and dev_RAT10 > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_RAT10
                            result_table_qr.add_row(
                                [epoch, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            result_table_tp.add_row(
                                [epoch, dev_PAT1, dev_PAT3, dev_PAT5, dev_PAT10, dev_RAT1, dev_RAT3, dev_RAT5, dev_RAT10,
                                 test_PAT1, test_PAT3, test_PAT5, test_PAT10, test_RAT1, test_RAT3, test_RAT5, test_RAT10]
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
        path = "{}_{}".format(path, ".pkl.gz")
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


class LstmQRTP(ModelQRTP):

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

            if self.args.normalize:
                self.t_states_series = self.normalize_3d(self.t_states_series)

            if self.args.average == 1:
                self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
            elif self.args.average == 0:
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


class BiRNNQRTP(ModelQRTP):

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


class GruQRTP(ModelQRTP):

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

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average == 1:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            elif self.args.average == 0:
                self.b_state = self.b_current_state[0]
            else:
                self.b_state = self.maximum_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)

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
