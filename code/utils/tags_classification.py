import tensorflow as tf
from statistics import read_df
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss
import numpy as np
from dio import bp_mll_loss
from prettytable import PrettyTable
import os
from qa import myio
import gzip
import pickle
from sklearn.model_selection import StratifiedShuffleSplit


class Model(object):

    def __init__(self, n_in, n_hid, n_out):

        self.input = tf.placeholder(tf.float32, [None, None], name='input')
        self.target = tf.placeholder(tf.float32, [None, None], name='target')

        with tf.name_scope('hidden_layer'):
            self.w_h = tf.Variable(tf.random_normal([n_in, n_hid], mean=0.0, stddev=0.05), name='w_h')
            self.b_h = tf.Variable(tf.zeros([n_hid]), name='b_h')

            hid_out = tf.matmul(self.input, self.w_h) + self.b_h
            self.hid_out = tf.nn.relu(hid_out)

        with tf.name_scope('output_layer'):
            self.w_o = tf.Variable(tf.random_normal([n_hid, n_out], mean=0.0, stddev=0.05), name='w_o')
            self.b_o = tf.Variable(tf.zeros([n_out]), name='b_o')

            out = tf.matmul(self.hid_out, self.w_o) + self.b_o
            self.out = out
            self.output = tf.nn.sigmoid(out)

        self.params = [self.w_h, self.b_h, self.w_o, self.b_o]

        with tf.name_scope('loss'):
            # modified cross entropy to explicit mathematical formula of sigmoid cross entropy loss
            loss1 = -tf.reduce_sum(
                (self.target * tf.log(self.output + 1e-9)) + ((1 - self.target) * tf.log(1 - self.output + 1e-9)),
                name='cross_entropy'
            )
            # loss2 = tf.reduce_sum(
            #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=self.output)
            # )  # should be same with loss1 but something is wrong and this doesnt decrease..
            self.loss = loss1

            # regularization
            l2 = (tf.nn.l2_loss(self.w_h) + tf.nn.l2_loss(self.w_o))
            lambda_2 = 0.01

            self.cost = self.loss + lambda_2 * l2

            # self.loss3 = bp_mll_loss(self.target, self.output)  # for some reason it gives nan ...

    @staticmethod
    def precision_recall_hamming(output, target, threshold):
        print 'output ', np.sum(output > 0.5)
        print 'target ', np.sum(target > 0.4)

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

        hl = hamming_loss(target, predictions)

        return round(pre, 4), round(rec, 4), round(hl, 4)

    @staticmethod
    def one_error(outputs, targets):
        cols = np.argmax(outputs, 1)  # top ranked
        rows = range(outputs.shape[0])
        result = targets[rows, cols]
        return np.sum((result == 0).astype(np.int32))

    def eval_batch(self, x_batch, y_batch, sess):
        # precision_recall_hamming_loss = []
        output = sess.run(
            self.output,
            feed_dict={
                self.input: x_batch.toarray(),
            }
        )
        # bp_mll = bp_mll_loss(y_batch.astype(np.float32), output)
        # for threshold in range(7, 2, -1):
        #     pre, rec, ham_loss = clf.precision_recall_hamming(output, y_batch, threshold * 0.1)
        #     precision_recall_hamming_loss.append((pre, rec, ham_loss))
        pre, rec, ham_loss = self.precision_recall_hamming(output, y_batch, self.threshold)
        oe = self.one_error(output, y_batch)
        return oe, pre, rec, ham_loss  # , bp_mll

    def evaluate(self, dev_batches, sess):
        oe = []
        pre = []
        rec = []
        ham_loss = []
        for x_b, y_b in dev_batches:
            batch_oe, batch_pre, batch_rec, batch_ham_loss = self.eval_batch(x_b, y_b, sess)
            oe.append(batch_oe), pre.append(batch_pre), rec.append(batch_rec), ham_loss.append(batch_ham_loss)
        oe = sum(oe) / len(oe)
        pre = sum(pre) / len(pre)
        rec = sum(rec) / len(rec)
        ham_loss = sum(ham_loss) / len(ham_loss)
        return oe, pre, rec, ham_loss

    def train_batch(self, x_batch, y_batch, train_op, global_step, train_summary_op, train_summary_writer, sess):
        _, _step, _loss, _cost, _summary = sess.run(
            [train_op, global_step, self.loss, self.cost, train_summary_op],
            feed_dict={
                self.input: x_batch.toarray(),
                self.target: y_batch
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

            print("Writing to {}\n".format(DIR))

            # Summaries for loss and cost
            loss_summary = tf.summary.scalar("loss", self.loss)
            cost_summary = tf.summary.scalar("cost", self.cost)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, cost_summary])
            train_summary_dir = os.path.join(DIR, "summaries", "train")
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
                dev_summary_dir = os.path.join(DIR, "summaries", "dev")
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
                test_summary_dir = os.path.join(DIR, "summaries", "test")
                test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            checkpoint_dir = os.path.join(DIR, "checkpoints")
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
                    x_b, y_b = train_batches[i]
                    cur_step, cur_loss, cur_cost = self.train_batch(
                        x_b, y_b, train_op, global_step, train_summary_op, train_summary_writer, sess
                    )

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))

                    if i == N-1:  # EVAL
                        if dev:
                            self.threshold = 0.5
                            dev_OE, dev_PRE, dev_REC, dev_HL = self.evaluate(dev, sess)
                            _dev_sum = sess.run(
                                dev_summary_op,
                                {dev_oe: dev_OE, dev_pre: dev_PRE, dev_rec: dev_REC, dev_hl: dev_HL}
                            )
                            dev_summary_writer.add_summary(_dev_sum, cur_step)

                        if test:
                            self.threshold = 0.5
                            test_OE, test_PRE, test_REC, test_HL = self.evaluate(test, sess)
                            _test_sum = sess.run(
                                test_summary_op,
                                {test_oe: test_OE, test_pre: test_PRE, test_rec: test_REC, test_hl: test_HL}
                            )
                            test_summary_writer.add_summary(_test_sum, cur_step)

                        if dev:
                            if dev_PRE > best_pre:
                                unchanged = 0
                                best_pre = dev_PRE
                                result_table.add_row(
                                    [epoch, dev_OE, dev_PRE, dev_REC, dev_HL, dev_BP_MLL, test_OE, test_PRE, test_REC, test_HL, test_BP_MLL]
                                )
                                # self.save(sess, checkpoint_prefix, cur_step)
                        else:
                            if test and (test_PRE > best_pre):
                                unchanged = 0
                                best_pre = test_PRE
                                result_table.add_row(
                                    [epoch, dev_OE, dev_PRE, dev_REC, dev_HL, dev_BP_MLL, test_OE, test_PRE, test_REC,
                                     test_HL, test_BP_MLL]
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


def get_train_test_dev(df, labels):
    data = {}
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, ngram_range=(1, 1))
    vectorizer.fit((df['title'] + u' ' + df['body']).values)
    for type_name, group_df in df.groupby('type'):
        data[type_name] = (
            vectorizer.transform((group_df['title'] + u' ' + group_df['body']).values),
            group_df[labels].values
        )
    return data['train'], data['dev'], data['test']


if __name__ == '__main__':

    df = read_df('data.csv')
    df = df.fillna(u'')
    WAY = 1
    NAME = 'all_selected_300.p'
    # DIR = '/home/christina/Documents/Thesis/models/askubuntu/all_selected_300_tf_sklearnsplit_crossentropy/'
    DIR = '/home/christina/Documents/Thesis/models/askubuntu/todelete'

    labels = pickle.load(open(NAME, 'rb'))

    if WAY == 1:
        df_x = df['title'] + u' ' + df['body']
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, ngram_range=(1, 1))
        data_x = vectorizer.fit_transform(df_x.values)
        df_y = df[labels]
        data_y = df_y.as_matrix()

        stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
        for train_index, test_index in stratified_split.split(data_x, data_y):
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
    else:
        train_data, dev_data, test_data = get_train_test_dev(df, labels)
        x_train, y_train = train_data
        x_dev, y_dev = dev_data
        x_test, y_test = test_data

    def get_batches(x, y, batch_size=40):
        last_idx = 0
        while last_idx < x.shape[0] - 1:
            end_idx = min(last_idx + batch_size, x.shape[0] - 1)
            x_ = x[last_idx:end_idx]
            y_ = y[last_idx:end_idx]
            last_idx = end_idx
            yield x_, y_

    clf = Model(x_train.shape[1], 200, y_train.shape[1])

    clf.train_model(
        list(get_batches(x_train, y_train, 50)),
        # dev=list(get_batches(x_dev, y_dev, 50)),
        test=list(get_batches(x_test, y_test, 50))
    )


