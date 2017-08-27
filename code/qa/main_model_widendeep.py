import tensorflow as tf
import numpy as np
from nn import get_activation_by_name
from prettytable import PrettyTable
from myio import say
from nn.initialization import random_init
from evaluation import Evaluation
import myio
import os


from main_model_nostate import Model as BasicModel
NUM_FEATURES = 4


class Model(BasicModel):

    def _initialize_graph(self):

        with tf.name_scope('DNN_COMPONENT'):

            with tf.name_scope('input'):
                self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
                self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
                self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

                self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

            with tf.name_scope('embeddings'):
                self.titles = tf.nn.embedding_lookup(self.embeddings, self.titles_words_ids_placeholder)
                self.titles = tf.nn.dropout(self.titles, 1.0 - self.dropout_prob)

                self.bodies = tf.nn.embedding_lookup(self.embeddings, self.bodies_words_ids_placeholder)
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
                    dtype=tf.float32
                )
                # current_state = last state of every layer in the network as an LSTMStateTuple

                if self.args.normalize:
                    self.t_states_series = self.normalize_3d(self.t_states_series)

                if self.args.average:
                    self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
                else:
                    self.t_state = self.t_states_series[:, -1, :]
                    # SAME AS self.t_current_state[-1][1]
                    # SAME AS self.t_current_state[0][1]

            with tf.name_scope('bodies_output'):
                self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(
                    cell,
                    self.bodies,
                    dtype=tf.float32
                )
                # current_state = last state of every layer in the network as an LSTMStateTuple

                if self.args.normalize:
                    self.b_states_series = self.normalize_3d(self.b_states_series)

                if self.args.average:
                    self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
                else:
                    self.b_state = self.b_states_series[:, -1, :]
                    # SAME AS self.b_current_state[-1][1]
                    # SAME AS self.b_current_state[0][1]

            with tf.name_scope('outputs'):
                # batch * d
                h_final = (self.t_state + self.b_state) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)

            with tf.name_scope('dnn-logits'):
                # For testing:
                #   first one in batch is query, the rest are candidate questions
                self.dnn_scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)

                # For training:
                pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
                # num query * n_d
                query_vecs = pairs_vecs[:, 0, :]
                # num query *num candidates
                self.dnn_logits = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 1:, :], axis=2)

        dnn_vars_to_train = set(tf.trainable_variables())

        with tf.name_scope('WIDE-COMPONENT'):

            with tf.name_scope("input"):
                # batch size * candidates * features
                self.features_placeholder = tf.placeholder(
                    tf.float32, [None, None, NUM_FEATURES], name='features'
                )

                # if self.args.word_vec_feature:
                #     self.summed_vectors = tf.reduce_sum(self.titles, axis=1) + tf.reduce_sum(self.bodies, axis=1)
                #     # For testing:
                #     self.sv_features = tf.reduce_sum(tf.multiply(self.summed_vectors[0], self.summed_vectors[1:]), axis=1)
                #     # For training:
                #     sv_features2 = tf.nn.embedding_lookup(
                #         self.summed_vectors, self.pairs_ids_placeholder, name='summed_vec_features'
                #     )
                #     self.sv_features2 = tf.reduce_sum(tf.expand_dims(sv_features2[:, 0, :], axis=1) * sv_features2[:, 1:, :], axis=2)
                #     self.features = tf.concat([self.features_placeholder, ])

            with tf.name_scope("weights"):
                self.W = tf.Variable(random_init([NUM_FEATURES, 1]), name="weight")
                self.b = tf.Variable(np.random.random(), name="bias")

            with tf.name_scope('outputs'):
                # Construct a linear model
                # one prediction for each sample in the batch and each candidate
                predictions = tf.matmul(tf.reshape(self.features_placeholder, [-1, NUM_FEATURES]), self.W)

            with tf.name_scope('linear-logits'):
                # For testing:
                #   first one in batch is query, the rest are candidate questions
                self.linear_scores = tf.add(predictions, self.b)

                self.linear_logits = self.linear_scores
                self.linear_logits = tf.reshape(self.linear_logits, [-1, 21])

        linear_vars_to_train = set(tf.trainable_variables()) - dnn_vars_to_train

        self.dnn_vars_to_train, self.linear_vars_to_train = list(dnn_vars_to_train), list(linear_vars_to_train)

        # for testing
        with tf.name_scope('WND-scores'):
            # in training we have one query, one positive and 20 negative
            # in testing we have one query, 20 candidates and one dummy
            self.scores = self.dnn_scores + tf.squeeze(self.linear_scores[:-1])

        with tf.name_scope('WND-logits'):
            self.logits = self.dnn_logits + self.linear_logits
            self.logits = self.normalize_2d(self.logits)  # todo: keep it? i think it does change anythin

        with tf.name_scope('cost'):

            pos_scores = self.logits[:, 0]
            neg_scores = self.logits[:, 1:]
            neg_scores = tf.reduce_max(neg_scores, axis=1)

            with tf.name_scope('loss'):

                delta = 1.0

                diff = neg_scores - pos_scores + delta
                loss = tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff)
                self.loss = loss

            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in self.dnn_vars_to_train + self.linear_vars_to_train:
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg

            self.cost = self.loss + self.l2_reg

    def _eval_batch(self, titles, bodies, features, sess):
        extended_features = np.zeros((21, features.shape[1]))
        extended_features[0:-1] = features  # we add a dummy instance at the end

        _scores = sess.run(
            self.scores,
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
                self.features_placeholder: np.reshape(extended_features, (1, 21, features.shape[1]))  # batch size 1
            }
        )

        return _scores

    def evaluate(self, data, sess):
        res = []
        hinge_loss = 0.

        for idts, idbs, id_labels, batch_features in data:
            cur_scores = self._eval_batch(idts, idbs, batch_features, sess)
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

    def _train_batch(self, titles, bodies, pairs, features, train_op, global_step, train_summary_op, train_summary_writer, sess):
        _, _step, _loss, _cost, _summary = sess.run(
            [train_op, global_step, self.loss, self.cost, train_summary_op],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.dropout_prob: np.float32(self.args.dropout),
                self.features_placeholder: features
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
                    idts, idbs, idps, batch_features = train_batches[i]
                    cur_step, cur_loss, cur_cost = self._train_batch(
                        idts, idbs, idps, batch_features, train_op, global_step, train_summary_op, train_summary_writer, sess
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

