import tensorflow as tf
import numpy as np
from evaluation import Evaluation
from nn import get_activation_by_name, init_w_b_vals
import gzip
import pickle
from prettytable import PrettyTable
from myio import say
import myio
import os
from losses import loss0, loss1, loss2, loss0sum, loss2sum
from losses import devloss0, devloss1, devloss2, devloss0sum, devloss2sum
from losses import dev_entropy_loss


class ModelQR(object):

    def ready(self):
        self._initialize_placeholders_graph()
        self._initialize_encoder_graph()
        self._initialize_output_graph()
        for param in tf.trainable_variables():
            self.params[param.name] = param
        self.params[self.embeddings.name] = self.embeddings  # in case it is not trainable

    def _initialize_placeholders_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

            self.query_per_pair = tf.placeholder(tf.int32, [None, None], name='query_per_pair')

            self.target_scores = tf.placeholder(tf.float32, [None, None], name='target_scores')

    def _initialize_output_graph(self):

        with tf.name_scope('scores'):
            # Note: Multiplying two dot vectors (cosine) gives a value in [-1, 1]
            # For testing:
            #   first one in batch is query, the rest are candidate questions
            self.scores = tf.reduce_sum(tf.multiply(self.h_final[0], self.h_final[1:]), axis=1)

            # scores in [-1, 1]

            # For training:
            pairs_vecs = tf.nn.embedding_lookup(self.h_final, self.pairs_ids_placeholder, name='pairs_vecs')
            # [num_of_tuples, n_d]
            query_vecs = pairs_vecs[:, 0, :]
            # [num_of_tuples]
            pos_scores = tf.reduce_sum(query_vecs * pairs_vecs[:, 1, :], axis=1)
            # [num_of_tuples, candidate size - 1]
            all_neg_scores = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 2:, :], axis=2)

            # self.pos_scores = pos_scores
            # self.all_neg_scores = all_neg_scores

        if 'mlp_dim' in self.args and self.args.mlp_dim != 0:
            raise Exception('unimplemented')  # todo
            self.mlp(query_vecs, pairs_vecs[:, 1:, :])

        with tf.name_scope('cost'):
            # h_final can have negative values, pos_scores and neg_scores have values in [0,1]

            with tf.name_scope('loss'):

                if 'entropy' not in self.args or self.args.entropy == 0:
                    if 'loss' in self.args and self.args.loss == 'loss1':
                        self.loss = loss1(pos_scores, all_neg_scores, self.query_per_pair)  # OK
                        self.loss1 = loss0(pos_scores, all_neg_scores)  # alternative 1
                        self.loss2 = loss2(pos_scores, all_neg_scores)  # alternative 2
                    elif 'loss' in self.args and self.args.loss == 'loss2':
                        self.loss = loss2(pos_scores, all_neg_scores)  # OK
                        self.loss1 = loss1(pos_scores, all_neg_scores, self.query_per_pair)  # alternative 1
                        self.loss2 = loss0(pos_scores, all_neg_scores)  # alternative 2
                    elif 'loss' in self.args and self.args.loss == 'loss2sum':
                        self.loss = loss2sum(pos_scores, all_neg_scores, self.query_per_pair)  # OK
                    elif 'loss' in self.args and self.args.loss == 'loss0sum':
                        self.loss = loss0sum(pos_scores, all_neg_scores, self.query_per_pair)  # OK
                    else:
                        self.loss = loss0(pos_scores, all_neg_scores)  # OK
                        self.loss1 = loss1(pos_scores, all_neg_scores, self.query_per_pair)  # alternative 1
                        self.loss2 = loss2(pos_scores, all_neg_scores)  # alternative 2
                else:
                    logits = tf.reduce_sum(tf.expand_dims(query_vecs, axis=1) * pairs_vecs[:, 1:, :], axis=2)

                    def normalized(v):
                        min_ = tf.reduce_min(v)
                        max_ = tf.reduce_max(v)
                        return (v - min_) / (max_ - min_)

                    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.target_scores,
                        logits=normalized(logits)
                    )
                    self.loss = tf.reduce_mean(tf.reduce_sum(x_entropy, axis=1), name='x_entropy')
                    self.loss1 = loss1(pos_scores, all_neg_scores, self.query_per_pair)  # alternative 1
                    self.loss2 = loss2(pos_scores, all_neg_scores)  # alternative 2

            with tf.name_scope('regularization'):
                l2_reg = 0.
                for param in set(tf.trainable_variables() + [self.embeddings]):  # in case not trainable emb
                    l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                self.l2_reg = l2_reg

            self.cost = self.args.weight*self.loss + self.l2_reg

    def mlp(self, queries_vec, candidates_vec):  # [None=tuples_num, hidden_dim], [None=tuples_num, 21, hidden_dim]

        # [None=tuples_num, 21, hidden_dim]
        q_exp = tf.tile(tf.reshape(queries_vec, (-1, 1, self.args.hidden_dim)), (1, 21, 1))
        # [None=tuples_num*2, 21, hidden_dim]
        a = tf.reshape(
            tf.concat([q_exp[:, ::2, :], candidates_vec[:, ::2, :]], axis=1),
            (-1, 21/2+1, self.args.hidden_dim*2)
        )
        b = tf.reshape(
            tf.concat([q_exp[:, 1::2, :], candidates_vec[:, 1::2, :]], axis=1),
            (-1, 21/2, self.args.hidden_dim*2)
        )
        mixed = tf.stack([a, b], axis=1)
        self.mixed1 = mixed
        # [None=tuples_num*21, hidden_dim*2]
        mixed = tf.reshape(mixed, (-1, self.args.hidden_dim*2))

        self.mixed2 = mixed
        return

        with tf.name_scope('MLP'):
            self.h = self.layer(mixed, self.args.hidden_dim*2, self.args.mlp_dim, 'HL1')
            self.o = self.layer(self.h, self.args.mlp_dim, 1, 'O')

    def layer(self, previous, hidden_dim_in, hidden_dim_out, name):
        w, b = init_w_b_vals([hidden_dim_in, hidden_dim_out], [hidden_dim_out], self.args.activation)
        weights, biases = tf.Variable(w, name='w{}'.format(name)), tf.Variable(b, name='b{}'.format(name))
        r = tf.add(tf.matmul(previous, weights), biases)
        r = tf.nn.relu(r)
        return r

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
        all_labels = []
        all_scores = []

        sample = 0
        for idts, idbs, id_labels in data:
            sample += 1
            cur_scores = self.eval_batch(idts, idbs, sess)
            assert len(id_labels) == len(cur_scores)  # equal to 20

            all_labels.append(id_labels)
            all_scores.append(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = Evaluation(res)
        MAP = e.MAP()
        MRR = e.MRR()
        P1 = e.Precision(1)
        P5 = e.Precision(5)
        if 'entropy' in self.args and self.args.entropy != 0:
            loss1 = dev_entropy_loss(all_labels, all_scores)
        else:
            loss1 = devloss1(all_labels, all_scores)
        if 'loss' in self.args and 'sum' in self.args.loss:
            loss0 = devloss0sum(all_labels, all_scores)
            loss2 = devloss2sum(all_labels, all_scores)
        else:
            loss0 = devloss0(all_labels, all_scores)
            loss2 = devloss2(all_labels, all_scores)
        return MAP, MRR, P1, P5, loss0, loss1, loss2

    def train_batch(self, titles, bodies, pairs, query_per_pair, train_op, global_step, sess):
        # _, _step, _loss, _cost, _pos_scores, _all_neg_scores, _loss1, _loss2 = sess.run(
        #     [train_op, global_step, self.loss, self.cost, self.pos_scores, self.all_neg_scores, self.loss1, self.loss2],
        target_scores = np.zeros((len(pairs), 21))
        target_scores[:, 0] = 1.
        _, _step, _loss, _cost, _loss1, _loss2 = sess.run(
            [train_op, global_step, self.loss, self.cost, self.loss1, self.loss2],
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.pairs_ids_placeholder: pairs,
                self.dropout_prob: np.float32(self.args.dropout),
                self.query_per_pair: query_per_pair,
                self.target_scores: target_scores
            }
        )
        # scores = np.hstack([_pos_scores.reshape([-1, 1]), _all_neg_scores])
        # ranks = (-scores).argsort()
        # ranked_labels = []
        # for rank in ranks:
        #     ranked_labels.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[rank])
        # ev = Evaluation(ranked_labels)
        # return _step, _loss, _cost, ev.MAP(), ev.MRR(), ev.Precision(1), ev.Precision(5)
        return _step, _loss, _cost, None, None, None, None, _loss1, _loss2

    def train_model(self, ids_corpus, train, dev=None, test=None):
        with tf.Session() as sess:

            result_table = PrettyTable(
                ["Epoch", "Step", "dev MAP", "dev MRR", "dev P@1", "dev P@5", "tst MAP", "tst MRR", "tst P@1", "tst P@5"]
            )
            dev_MAP = dev_MRR = dev_P1 = dev_P5 = 0
            test_MAP = test_MRR = test_P1 = test_P5 = 0
            best_dev = -1

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
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
            train_loss_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "loss"),
            )
            train_cost_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "cost"), sess.graph
            )

            train_loss1_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "loss1"),
            )
            train_loss2_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "train", "loss2"),
            )

            # train_map_writer = tf.summary.FileWriter(
            #     os.path.join(self.args.save_dir, "summaries", "train", "map"),
            # )
            # train_mrr_writer = tf.summary.FileWriter(
            #     os.path.join(self.args.save_dir, "summaries", "train", "mrr"),
            # )
            # train_pat1_writer = tf.summary.FileWriter(
            #     os.path.join(self.args.save_dir, "summaries", "train", "pat1"),
            # )
            # train_pat5_writer = tf.summary.FileWriter(
            #     os.path.join(self.args.save_dir, "summaries", "train", "pat5"),
            # )

            # VARIABLE NORM
            p_norm_summaries = {}
            p_norm_placeholders = {}
            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                p_norm_placeholders[param_name] = tf.placeholder(tf.float32)
                p_norm_summaries[param_name] = tf.summary.scalar(param_name, p_norm_placeholders[param_name])
            p_norm_summary_op = tf.summary.merge(p_norm_summaries.values())
            p_norm_summary_dir = os.path.join(self.args.save_dir, "summaries", "p_norm")
            p_norm_summary_writer = tf.summary.FileWriter(p_norm_summary_dir,)

            # DEV LOSS & EVAL
            dev_loss0_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "loss0"),
            )
            dev_loss1_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "loss1"),
            )
            dev_loss2_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "loss2"),
            )
            dev_eval_writer1 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "MAP"),
            )
            dev_eval_writer2 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "MRR"),
            )
            dev_eval_writer3 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Pat1"),
            )
            dev_eval_writer4 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Pat5"),
            )

            loss = tf.placeholder(tf.float32)
            loss_summary = tf.summary.scalar("loss", loss)
            dev_eval = tf.placeholder(tf.float32)
            dev_summary = tf.summary.scalar("QR_evaluation", dev_eval)
            cost = tf.placeholder(tf.float32)
            cost_summary = tf.summary.scalar("cost", cost)
            # train_eval = tf.placeholder(tf.float32)
            # train_summary = tf.summary.scalar("QR_train", train_eval)

            if self.args.save_dir != "":
                checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

            patience = 8 if 'patience' not in self.args else self.args.patience
            unchanged = 0
            max_epoch = self.args.max_epoch
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > patience:
                    break

                train_batches = myio.create_batches(
                    ids_corpus, train, self.args.batch_size, self.padding_id, pad_left=False
                )

                N = len(train_batches)

                train_loss = 0.0
                train_cost = 0.0

                for i in xrange(N):
                    idts, idbs, idps, qpp = train_batches[i]
                    cur_step, cur_loss, cur_cost, curmap, curmrr, curpat1, curpat5, curloss1, curloss2 = self.train_batch(
                        idts, idbs, idps, qpp, train_op, global_step, sess
                    )
                    summary = sess.run(loss_summary, {loss: cur_loss})
                    train_loss_writer.add_summary(summary, cur_step)
                    train_loss_writer.flush()
                    summary = sess.run(cost_summary, {cost: cur_cost})
                    train_cost_writer.add_summary(summary, cur_step)
                    train_cost_writer.flush()

                    summary = sess.run(loss_summary, {loss: curloss1})
                    train_loss1_writer.add_summary(summary, cur_step)
                    train_loss1_writer.flush()
                    summary = sess.run(loss_summary, {loss: curloss2})
                    train_loss2_writer.add_summary(summary, cur_step)
                    train_loss2_writer.flush()

                    # summary = sess.run(train_summary, {train_eval: curmap})
                    # train_map_writer.add_summary(summary, cur_step)
                    # train_map_writer.flush()
                    # summary = sess.run(train_summary, {train_eval: curmrr})
                    # train_mrr_writer.add_summary(summary, cur_step)
                    # train_mrr_writer.flush()
                    # summary = sess.run(train_summary, {train_eval: curpat1})
                    # train_pat1_writer.add_summary(summary, cur_step)
                    # train_pat1_writer.flush()
                    # summary = sess.run(train_summary, {train_eval: curpat5})
                    # train_pat5_writer.add_summary(summary, cur_step)
                    # train_pat5_writer.flush()

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        say("\r{}/{}".format(i, N))

                    if i == N-1 or (i % 10 == 0 and self.args.testing):  # EVAL
                        if dev:
                            dev_MAP, dev_MRR, dev_P1, dev_P5, dloss0, dloss1, dloss2 = self.evaluate(dev, sess)

                            summary = sess.run(loss_summary, {loss: dloss0})
                            dev_loss0_writer.add_summary(summary, cur_step)
                            dev_loss0_writer.flush()
                            summary = sess.run(loss_summary, {loss: dloss1})
                            dev_loss1_writer.add_summary(summary, cur_step)
                            dev_loss1_writer.flush()
                            summary = sess.run(loss_summary, {loss: dloss2})
                            dev_loss2_writer.add_summary(summary, cur_step)
                            dev_loss2_writer.flush()

                            summary = sess.run(dev_summary, {dev_eval: dev_MAP})
                            dev_eval_writer1.add_summary(summary, cur_step)
                            dev_eval_writer1.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_MRR})
                            dev_eval_writer2.add_summary(summary, cur_step)
                            dev_eval_writer2.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_P1})
                            dev_eval_writer3.add_summary(summary, cur_step)
                            dev_eval_writer3.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_P5})
                            dev_eval_writer4.add_summary(summary, cur_step)
                            dev_eval_writer4.flush()

                            feed_dict = {}
                            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                                feed_dict[p_norm_placeholders[param_name]] = param_norm
                            _p_norm_sum = sess.run(p_norm_summary_op, feed_dict)
                            p_norm_summary_writer.add_summary(_p_norm_sum, cur_step)

                        if test:
                            test_MAP, test_MRR, test_P1, test_P5, tloss0, tloss1, tloss2 = self.evaluate(test, sess)

                        if self.args.performance == "MRR" and dev_MRR > best_dev:
                            unchanged = 0
                            best_dev = dev_MRR
                            result_table.add_row(
                                [epoch, cur_step, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)
                        elif self.args.performance == "MAP" and dev_MAP > best_dev:
                            unchanged = 0
                            best_dev = dev_MAP
                            result_table.add_row(
                                [epoch, cur_step, dev_MAP, dev_MRR, dev_P1, dev_P5, test_MAP, test_MRR, test_P1, test_P5]
                            )
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        say("\r\n\nEpoch {}\tcost={:.3f}\tloss={:.3f}\tMRR={:.2f},MAP={:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss / (i+1),  # i.e. divided by N training batches
                            dev_MRR,
                            dev_MAP
                        ))
                        say("\n{}\n".format(result_table))
                        myio.say("\tp_norm: {}\n".format(
                            self.get_pnorm_stat(sess)
                        ))

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


class LstmQR(ModelQR):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
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
            # current_state = last state of every layer in the network as an LSTMStateTuple

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
            # current_state = last state of every layer in the network as an LSTMStateTuple

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average == 1:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            elif self.args.average == 0:
                # self.b_state=self.b_states_series[:, -1, :]=self.b_current_state[-1][1]=self.b_current_state[0][1]
                # in case sequence_length parameter is used in RNN, the last state is not self.b_states_series[:,-1,:]
                # but is self.b_states_series[:, self.SLB[x], :] and it is stored correctly in
                # self.b_current_state[0][1] so its better and safer to use this.
                self.b_state = self.b_current_state[0][1]
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


class BiRNNQR(ModelQR):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
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
            # forw_t_outputs = a Tensor shaped: [batch_size, max_time, cell_fw.output_size]
            # back_t_outputs = a Tensor shaped: [batch_size, max_time, cell_bw.output_size].
            # if BiLSTM
            # t_state: A tuple (output_state_fw, output_state_bw)
            # where each state is a tuple of the hidden and output states that are ndarrays of shape
            # [batch_size, cell_fw.output_size]
            # if BiGRU
            # t_state: A tuple (output_state_fw, output_state_bw)
            # where each state is an ndarray of shape [batch_size, cell_fw.output_size]

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
            # forw_t_outputs = a Tensor shaped: [batch_size, max_time, cell_fw.output_size]
            # back_t_outputs = a Tensor shaped: [batch_size, max_time, cell_bw.output_size].
            # if BiLSTM
            # t_state: A tuple (output_state_fw, output_state_bw)
            # where each state is a tuple of the hidden and output states that are ndarrays of shape
            # [batch_size, cell_fw.output_size]
            # if BiGRU
            # t_state: A tuple (output_state_fw, output_state_bw)
            # where each state is an ndarray of shape [batch_size, cell_fw.output_size]

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


class CnnQR(ModelQR):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
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
            # TensorFlow's convolutional conv2d operation expects a 4-dimensional
            # tensor with dimensions corresponding to batch, width, height and channel
            # The result of our embedding doesn't contain the channel dimension,
            # so we add it manually, leaving us with a layer of
            # shape [batch/None, sequence_length, embedding_size, 1].

            # So, each element of the word vector is a list of size 1 instead of a real number.

            # CONVOLUTION AND MAXPOOLING
            pooled_outputs_t = []
            pooled_outputs_b = []
            filter_sizes = [3]
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_layer.n_d, 1, self.args.hidden_dim]
                    print 'assuming num filters = hidden dim. IS IT CORRECT? '

                    w_vals, b_vals = init_w_b_vals(filter_shape, [self.args.hidden_dim], self.args.activation)
                    W = tf.Variable(w_vals, name="conv-W")
                    b = tf.Variable(b_vals, name="conv-b")
                    # self.W = W

                    with tf.name_scope('titles_output'):
                        conv_t = tf.nn.conv2d(
                            self.embedded_titles_expanded,
                            W,
                            strides=[1, 1, 1, 1],  # how much the window shifts by in each of the dimensions.
                            padding="VALID",
                            name="conv-titles")

                        # Apply nonlinearity
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

                        # self.pooled_t = pooled_t
                        pooled_outputs_t.append(pooled_t)

                    with tf.name_scope('bodies_output'):
                        conv_b = tf.nn.conv2d(
                            self.embedded_bodies_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv-bodies")
                        # self.conv_b = conv_b

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

                        # self.pooled_b = pooled_b
                        pooled_outputs_b.append(pooled_b)

            # Combine all the pooled features
            num_filters_total = self.args.hidden_dim * len(filter_sizes)
            self.t_pool = tf.concat(pooled_outputs_t, 3)
            self.t_state = tf.reshape(self.t_pool, [-1, num_filters_total])
            # reshape so that we have shape [batch, num_features_total]

            self.b_pool = tf.concat(pooled_outputs_b, 3)
            self.b_state = tf.reshape(self.b_pool, [-1, num_filters_total])

        with tf.name_scope('outputs'):
            # batch * d
            h_final = (self.t_state + self.b_state) * 0.5
            h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
            self.h_final = self.normalize_2d(h_final)


class GruQR(ModelQR):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
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
