import argparse
import os
import pickle
import random
import sys
import time

import numpy as np
import tensorflow as tf
from prettytable import PrettyTable

import tags_prediction.myio as myio_
from nn import get_activation_by_name
from qa import myio
from utils import load_embedding_iterator
from utils.statistics import read_df


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

        with tf.name_scope('LSTM'):

            def lstm_cell():
                _cell = tf.nn.rnn_cell.LSTMCell(
                    self.args.hidden_dim, state_is_tuple=True, activation=get_activation_by_name(self.args.activation)
                )
                return _cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(self.args.depth)]
            )

        with tf.name_scope('titles_output'):
            self.t_states_series, self.t_current_state = tf.nn.dynamic_rnn(cell, self.titles, dtype=tf.float32)

            if self.args.normalize:
                self.t_states_series = self.normalize_3d(self.t_states_series)

            if self.args.average:
                self.t_state = self.average_without_padding(self.t_states_series, self.titles_words_ids_placeholder)
            else:
                self.t_state = self.t_states_series[:, -1, :]

        with tf.name_scope('bodies_output'):
            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(cell, self.bodies, dtype=tf.float32)

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            else:
                self.b_state = self.b_states_series[:, -1, :]

        with tf.name_scope('outputs'):
            # batch * d
            h_final = (self.t_state + self.b_state) * 0.5
            self.h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
            self.h_final = self.normalize_2d(self.h_final)

            self.w_o = tf.Variable(
                tf.random_normal([self.args.hidden_dim, self.output_dim], mean=0.0, stddev=0.05),
                name='weights_out'
            )
            self.b_o = tf.Variable(tf.zeros([self.output_dim]), name='bias_out')

            out = tf.matmul(self.h_final, self.w_o) + self.b_o
            self.output = tf.nn.sigmoid(out)

            with tf.name_scope('cost'):
                with tf.name_scope('loss'):

                    if self.args.loss_type == 'xentropy':
                        self.loss = -tf.reduce_sum(
                            (self.target * tf.log(self.output + 1e-9)) + (
                            (1 - self.target) * tf.log(1 - self.output + 1e-9)),
                            name='cross_entropy'
                        )

                    else:
                        # todo: check why gives nan
                        # get true and false labels
                        shape = tf.shape(self.target)
                        y_i = tf.equal(self.target, tf.ones(shape))
                        y_i_bar = tf.not_equal(self.target, tf.ones(shape))

                        # get indices to check
                        truth_matrix = tf.to_float(tf.logical_and(tf.expand_dims(y_i, 2), tf.expand_dims(y_i_bar, 1)))

                        # calculate all exp'd differences
                        sub_matrix = tf.subtract(tf.expand_dims(self.output, 2), tf.expand_dims(self.output, 1))

                        exp_matrix = tf.exp(tf.negative(sub_matrix))

                        # check which differences to consider and sum them
                        sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
                        sums = tf.reduce_sum(sparse_matrix, axis=[1, 2])

                        # get normalizing terms and apply them
                        y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
                        y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_i_bar), axis=1)
                        normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
                        results = tf.divide(sums, normalizers)

                        # sum over samples
                        self.loss = tf.reduce_sum(results)

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

    @staticmethod
    def precision_recall(output, target, threshold):

        #todo: if threshold = 0.5:
        # correct_prediction = np.equal(np.round(output), target)
        # accuracy = np.mean(correct_prediction)

        # print 'output ', np.sum(output > 0.5)
        # print 'target ', np.sum(target > 0.4)

        predictions = np.where(output > threshold, np.ones_like(output), np.zeros_like(output))

        # true positives
        tp = np.logical_and(predictions.astype(np.bool), target.astype(np.bool))
        # false positives
        fp = np.logical_and(predictions.astype(np.bool), np.logical_not(target.astype(np.bool)))
        # false negatives
        fn = np.logical_and(np.logical_not(predictions.astype(np.bool)), target.astype(np.bool))

        if np.sum(np.logical_or(tp, fp).astype(np.int32)) == 0:
            print 'zero here'
        if np.sum(np.logical_or(tp, fn).astype(np.int32)) == 0:
            print 'zero there'
        pre = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fp).astype(np.int32)))
        rec = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fn).astype(np.int32)))

        return round(pre, 4), round(rec, 4)

    @staticmethod
    def one_error(outputs, targets):
        cols = np.argmax(outputs, 1)  # top ranked
        rows = range(outputs.shape[0])
        result = targets[rows, cols]
        return np.sum((result == 0).astype(np.int32))

    @staticmethod  # todo: check it
    def precision_at_k(outputs, targets, k):
        cols = np.argsort(outputs, 1)[:, :k]
        rows = range(outputs.shape[0])
        rel_per_sample = np.clip(np.sum(targets, 1), 0, k)
        found_per_sample = np.zeros(outputs.shape[0])
        for sample, c in enumerate(cols):
            result = targets[rows, cols]
            found_per_sample[sample] = np.sum(result == 1)
        return np.mean(found_per_sample/rel_per_sample.astype(np.float32))

    def eval_batch(self, titles, bodies, y_batch, sess):
        output = sess.run(
            self.output,
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        pre, rec = self.precision_recall(output, y_batch, self.args.threshold)
        oe = self.one_error(output, y_batch)
        return oe, pre, rec

    def evaluate(self, dev_batches, sess):
        oe = []
        pre = []
        rec = []
        for titles_b, bodies_b, tags_b in dev_batches:
            batch_oe, batch_pre, batch_rec = self.eval_batch(titles_b, bodies_b, tags_b, sess)
            oe.append(batch_oe), pre.append(batch_pre), rec.append(batch_rec)
        oe = sum(oe) / len(oe)
        pre = sum(pre) / len(pre)
        rec = sum(rec) / len(rec)
        return oe, pre, rec, 0

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
                ["Epoch", "dev OE", "dev PRE", "dev REC", "dev HL", "dev BP-MLL", "tst OE", "tst PRE", "tst REC", "tst HL", "tst BP-MLL"]
            )

            dev_OE = dev_PRE = dev_REC = dev_HL = dev_BP_MLL = 0
            test_OE = test_PRE = test_REC = test_HL = test_BP_MLL = 0
            best_pre = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)
            grads_and_vars = optimizer.compute_gradients(self.cost)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            if assign_ops:
                print 'assigning trained values ...\n'
                sess.run(assign_ops)

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
                dev_oe = tf.placeholder(tf.float32)
                dev_pre = tf.placeholder(tf.float32)
                dev_rec = tf.placeholder(tf.float32)
                dev_hl = tf.placeholder(tf.float32)
                # dev_bp_mll = tf.placeholder(tf.float32)
                dev_oe_summary = tf.summary.scalar("dev_oe", dev_oe)
                dev_pre_summary = tf.summary.scalar("dev_pre", dev_pre)
                dev_rec_summary = tf.summary.scalar("dev_rec", dev_rec)
                dev_hl_summary = tf.summary.scalar("dev_hl", dev_hl)
                # dev_bp_mll_summary = tf.summary.scalar("dev_bp_mll", dev_bp_mll)
                dev_summary_op = tf.summary.merge(
                    [dev_oe_summary, dev_pre_summary, dev_rec_summary, dev_hl_summary]#, dev_bp_mll_summary]
                )
                dev_summary_dir = os.path.join(self.args.save_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            if test:
                # Test Summaries
                test_oe = tf.placeholder(tf.float32)
                test_pre = tf.placeholder(tf.float32)
                test_rec = tf.placeholder(tf.float32)
                test_hl = tf.placeholder(tf.float32)
                # test_bp_mll = tf.placeholder(tf.float32)
                test_oe_summary = tf.summary.scalar("test_oe", test_oe)
                test_pre_summary = tf.summary.scalar("test_pre", test_pre)
                test_rec_summary = tf.summary.scalar("test_rec", test_rec)
                test_hl_summary = tf.summary.scalar("test_hl", test_hl)
                # dev_bp_mll_summary = tf.summary.scalar("test_bp_mll", test_bp_mll)
                test_summary_op = tf.summary.merge(
                    [test_oe_summary, test_pre_summary, test_rec_summary, test_hl_summary]#, test_bp_mll_summary]
                )
                test_summary_dir = os.path.join(self.args.save_dir, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
            # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
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
                        titles_b, bodies_b, tag_labels_b, train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))

                    if i == N-1:  # EVAL
                        if dev:
                            dev_OE, dev_PRE, dev_REC, dev_HL = self.evaluate(dev, sess)
                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_oe: dev_OE, dev_pre: dev_PRE, dev_rec: dev_REC, dev_hl: dev_HL}
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            test_OE, test_PRE, test_REC, test_HL = self.evaluate(test, sess)
                            _test_sum = sess.run(
                                test_summary_op,
                                {test_oe: test_OE, test_pre: test_PRE, test_rec: test_REC, test_hl: test_HL}
                            )
                            test_summary_writer.add_summary(_test_sum, cur_step)

                        if dev_PRE > best_pre:
                            unchanged = 0
                            best_pre = dev_PRE
                            result_table.add_row(
                                [epoch, dev_OE, dev_PRE, dev_REC, dev_HL, dev_BP_MLL, test_OE, test_PRE, test_REC, test_HL, test_BP_MLL]
                            )
                            # self.save(sess, checkpoint_prefix, cur_step)

                        myio.say("\r\n\nEpoch {}\tcost={:.3f}\tloss={:.3f}\tPRE={:.2f},{:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss / (i+1),  # i.e. divided by N training batches
                            dev_PRE,
                            best_pre
                        ))
                        myio.say("\n{}\n".format(result_table))


def main(args):
    df = read_df(args.df_path)
    df = df.fillna(u'')

    label_tags = pickle.load(open(args.tags_file, 'rb'))

    raw_corpus = myio_.read_corpus(args.corpus, with_tags=True)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d=240,
                cut_off=1,
                embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    ids_corpus_tags = myio_.make_tag_labels(df, label_tags)

    ids_corpus = myio_.map_corpus(raw_corpus, embedding_layer, ids_corpus_tags, max_len=args.max_seq_len)

    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        weights = myio_.create_idf_weights(args.corpus, embedding_layer, with_tags=True)

    dev = myio_.create_batches(df, ids_corpus, 'dev', args.batch_size, padding_id, pad_left=not args.average)
    test = myio_.create_batches(df, ids_corpus, 'test', args.batch_size, padding_id, pad_left=not args.average)
    train = myio_.create_batches(df, ids_corpus, 'train', args.batch_size, padding_id, pad_left=not args.average)
    print '{} batches of {} instances in dev, {} in test and {} in train.'.format(
        len(dev), args.batch_size, len(test), len(train))

    model = Model(args, embedding_layer, weights=weights if args.reweight else None)
    model.ready()

    model.train_model(train, dev=dev, test=test)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--df_path", type=str)

    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)

    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--reweight", type=int, default=1)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--depth", type=int, default=1)

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)
    argparser.add_argument("--tags_file", type=str)

    argparser.add_argument("--loss_type", type=str, default='xentropy')
    argparser.add_argument("--threshold", type=float, default=0.5)

    args = argparser.parse_args()
    print args
    print
    main(args)

