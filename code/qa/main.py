import argparse
import myio
import sys
from utils import load_embedding_iterator
import time
import tensorflow as tf
import numpy as np
import os


class Model(object):

    def __init__(self, args, embedding_layer):
        self.args = args
        self.embeddings = embedding_layer.embeddings
        self._initialize_graph()

    def _initialize_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

            self.init_state = tf.placeholder(tf.float32, [self.args.depth, 2, None, self.args.hidden_dim], name='init_state')

        with tf.name_scope('embeddings'):
            self.titles = tf.nn.embedding_lookup(self.embeddings, self.titles_words_ids_placeholder)
            self.titles = tf.nn.dropout(self.titles, 1.0 - self.dropout_prob)

            self.bodies = tf.nn.embedding_lookup(self.embeddings, self.bodies_words_ids_placeholder)
            self.bodies = tf.nn.dropout(self.bodies, 1.0 - self.dropout_prob)

        with tf.name_scope('LSTM'):
            state_per_layer_list = tf.unstack(self.init_state, axis=0)  # i.e. unstack for each layer
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])  # i.e. cell_s, hidden_s
                 for idx in range(self.args.depth)]
            )

            def lstm_cell():
                _cell = tf.nn.rnn_cell.LSTMCell(self.args.hidden_dim, state_is_tuple=True)
                # _cell = tf.nn.rnn_cell.DropoutWrapper(_cell, output_keep_prob=0.5)
                return _cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell() for _ in range(self.args.depth)]
            )

        with tf.name_scope('titles_output'):
            self.t_states_series, self.t_current_state = tf.nn.dynamic_rnn(cell, self.titles, initial_state=rnn_tuple_state)
            # current_state = last state of every layer in the network as an LSTMStateTuple
            # if self.normalize:
            #     self.t_state = self.normalize_3d(self.t_state)
            if self.args.average:
                self.t_state = self.t_current_state[-1][1]  # todo the average
            else:
                self.t_state = self.t_current_state[-1][1]

        with tf.name_scope('bodies_output'):
            self.b_states_series, self.b_current_state = tf.nn.dynamic_rnn(cell, self.bodies, initial_state=rnn_tuple_state)
            # current_state = last state of every layer in the network as an LSTMStateTuple
            # if self.normalize:
            #     self.b_state = self.normalize_3d(self.b_state)
            if self.args.average:
                self.b_state = self.b_current_state[-1][1]  # todo the average
            else:
                self.b_state = self.b_current_state[-1][1]

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

        with tf.name_scope('loss'):
            diff = neg_scores - pos_scores + 1.0
            loss = tf.reduce_mean(tf.cast((diff > 0), tf.float32) * diff)
            self.loss = loss

            # todo: calculate cost

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

    def train_model(self, train_batches, dev=None, test=None):
        with tf.Session() as sess:

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())

            print("Writing to {}\n".format(args.save_dir))
            # Train Summaries
            train_summary_op = tf.summary.scalar("loss", self.loss)
            train_summary_dir = os.path.join(args.save_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

            unchanged = 0
            max_epoch = args.max_epoch
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > 15:
                    break

                N = len(train_batches)

                train_loss = 0.0
                # todo cost

                def train_batch(titles, bodies, pairs):
                    _current_state = np.zeros((args.depth, 2, titles.T.shape[0], args.hidden_dim))  # CURRENT_STATE DEPENDS ON BATCH

                    _, step, _loss, _summary = sess.run(
                        [train_op, global_step, self.loss, train_summary_op],
                        feed_dict={
                            self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                            self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                            self.pairs_ids_placeholder: pairs,
                            self.dropout_prob: np.float32(args.dropout),
                            self.init_state: _current_state
                        }
                    )

                    return _loss

                def eval_batch(titles, bodies, pairs):
                    _current_state = np.zeros((args.depth, 2, titles.T.shape[0], args.hidden_dim))  # CURRENT_STATE DEPENDS ON BATCH

                    _, step, _loss, _summary = sess.run(
                        [train_op, global_step, self.loss, train_summary_op],
                        feed_dict={
                            self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                            self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                            self.pairs_ids_placeholder: pairs,
                            self.dropout_prob: 0.,
                            self.init_state: _current_state
                        }
                    )

                    return _loss

                for i in xrange(N):
                    idts, idbs, idps = train_batches[i]
                    cur_loss = train_batch(idts, idbs, idps)
                    print '  loss ', cur_loss

                    train_loss += cur_loss
                    # todo cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))


def main(args):
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d=args.hidden_dim,
                cut_off=args.cut_off,
                embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
            )
    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)

    if args.train:
        start_time = time.time()
        train = myio.read_annotations(args.train)
        train_batches = myio.create_batches(
            ids_corpus, train, args.batch_size, padding_id, pad_left=not args.average
        )
        print("{} to create batches\n".format(time.time()-start_time))
        print("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))

        model = Model(args, embedding_layer)
        model.train_model(train_batches)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")

    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)

    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--reweight", type=int, default=1)

    argparser.add_argument("--load_pretrain", type=str, default="")

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print args
    print ""
    main(args)
