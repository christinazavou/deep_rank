import os
import pickle
import gzip
import numpy as np
import tensorflow as tf
from prettytable import PrettyTable
import myio
from nn import get_activation_by_name, init_w_b_vals
from evaluation import Evaluation
from losses import entropy_loss, dev_hinge_loss, dev_entropy_loss, hinge_loss
from losses import modified_hinge_loss, dev_modified_hinge_loss


class ModelMultiTagsClassifier(object):

    def ready(self):
        self._initialize_placeholders_graph()
        self._initialize_encoder_graph()
        for param in tf.trainable_variables():
            self.encoder_params.append(param.name)
        self._initialize_output_graph()
        for param in tf.trainable_variables():
            self.params[param.name] = param
        self.params[self.embeddings.name] = self.embeddings  # in case it is not trainable

    def _initialize_placeholders_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.target = tf.placeholder(tf.float32, [None, self.output_dim], name='target_tags')

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

            self.tuples = tf.placeholder(tf.int32, [None, None], name='tuples')

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

    def _initialize_output_graph(self):

        with tf.name_scope('outputs'):

            if self.args.mlp_dim != 0:
                scope_name = 'MLP'
            else:
                scope_name = 'SLP'

            with tf.name_scope(scope_name):

                if self.args.mlp_dim != 0:
                    w_h1, b_h1 = init_w_b_vals(
                        [self.args.hidden_dim, self.args.mlp_dim], [self.args.mlp_dim], self.args.activation)
                    weights_h1, biases_h1 = tf.Variable(w_h1, name='weights_h1'), tf.Variable(b_h1, name='bias_h1')
                    layer_1 = tf.add(tf.matmul(self.h_final, weights_h1), biases_h1)

                    # without activation is equivalent to SLP i.e. non linear
                    # act_layer_1 = get_activation_by_name(self.args.activation)(layer_1)
                    act_layer_1 = tf.nn.relu(layer_1)  # to reduce training time

                    w_o, b_o = init_w_b_vals([self.args.mlp_dim, self.output_dim], [self.output_dim], self.args.activation)
                    weights_o, biases_o = tf.Variable(w_o, name='weights_o'), tf.Variable(b_o, name='bias_o')

                    output = tf.matmul(act_layer_1, weights_o) + biases_o

                else:
                    w_o, b_o = init_w_b_vals([self.args.hidden_dim, self.output_dim], [self.output_dim], self.args.activation)
                    weights_o, biases_o = tf.Variable(w_o, name='weights_o'), tf.Variable(b_o, name='bias_o')

                    output = tf.matmul(self.h_final, weights_o) + biases_o

                self.b_o = biases_o  # for api compatibility

                self.act_output = tf.nn.sigmoid(output)

            # self.prediction = tf.where(
            #     self.act_output > self.args.threshold, tf.ones_like(self.act_output), tf.zeros_like(self.act_output)
            # )

            with tf.name_scope('cost'):
                # h_final and output can have negative values, act_output has values in [0,1]

                with tf.name_scope('loss'):
                    if 'entropy' not in self.args or self.args.entropy == 1:
                        self.loss = entropy_loss(self.args, self.target, self.act_output)
                    else:
                        if 'loss' in self.args and self.args.loss == 'loss0':
                            self.loss = hinge_loss(self.target, self.act_output, self.tuples, take_max=True)
                        else:
                            # self.loss = modified_hinge_loss(self.target, self.act_output, self.tuples)
                            self.loss = hinge_loss(self.target, self.act_output, self.tuples)

                with tf.name_scope('regularization'):
                    l2_reg = 0.
                    for param in set(tf.trainable_variables() + [self.embeddings]):  # in case not trainable emb
                        l2_reg += tf.nn.l2_loss(param) * self.args.l2_reg
                    self.l2_reg = l2_reg

                self.cost = self.loss + self.l2_reg

    def eval_batch(self, titles, bodies, sess):
        outputs = sess.run(
            self.act_output,
            feed_dict={
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: 0.,
            }
        )
        return outputs

    def evaluate(self, eval_batches, sess):

        outputs, targets = [], []
        tuples = []
        for titles_b, bodies_b, tags_b, tuples_b in eval_batches:
            out = self.eval_batch(titles_b, bodies_b, sess)
            outputs.append(out)
            targets.append(tags_b)
            tuples.append(tuples_b)

        outputs = np.vstack(outputs)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object
        tuples = np.vstack(tuples)

        if 'entropy' not in self.args or self.args.entropy == 1:
            loss = dev_entropy_loss(self.args, targets, outputs)
        else:
            if 'loss' in self.args and self.loss == 'loss0':
                loss = dev_hinge_loss(targets, outputs, tuples, take_max=True)
            else:
                loss = dev_modified_hinge_loss(targets, outputs, tuples)
                # loss = dev_hinge_loss(targets, outputs, tuples)

        """------------------------------------------remove ill evaluation-------------------------------------------"""
        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, targets = outputs[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""

        ev = Evaluation(outputs, None, targets)
        results = [ev.Precision(5), ev.Precision(10), ev.Recall(5), ev.Recall(10), ev.MeanAveragePrecision()]

        return loss, tuple(results)

    def train_batch(self, titles, bodies, y_batch, tuples, train_op, global_step, sess):
        _, _step, _loss, _cost = sess.run(
            [train_op, global_step, self.loss, self.cost],
            feed_dict={
                self.target: y_batch,
                self.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.dropout_prob: np.float32(self.args.dropout),
                self.tuples: tuples
            }
        )
        return _step, _loss, _cost

    def train_model(self, df, ids_corpus, dev=None, test=None):
        with tf.Session() as sess:

            result_table = PrettyTable(
                ["Epoch", "Step", "dev P@5", "dev P@10", "dev R@5", "dev R@10", "dev MAP",
                 "tst P@5", "tst P@10", "tst R@5", "tst R@10", "tst MAP"]
            )

            dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_MAP = 0, 0, 0, 0, 0
            test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_MAP = 0, 0, 0, 0, 0

            best_dev_performance = -1

            # Fine Encoder's params (and embeddings)
            non_encoder_params = []
            for param_name, param in self.params.iteritems():
                if param_name not in self.encoder_params:
                    non_encoder_params.append(param)
            print '\nnon encoder params {}\n'.format(non_encoder_params)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.args.optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(self.args.learning_rate)
            elif self.args.optimizer == "adagrad":
                optimizer = tf.train.AdagradOptimizer(self.args.learning_rate)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.args.learning_rate)

            if self.args.trainable_encoder:
                if self.args.activation in ['relu', 'lrelu']:
                    raise Warning('unimplemented gradient clipping')
                    # grads_and_vars = optimizer.compute_gradients(self.cost)
                    # clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
                    # train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)
                train_op = optimizer.minimize(self.cost, global_step=global_step)
            else:
                if self.args.activation in ['relu', 'lrelu']:
                    raise Warning('unimplemented gradient clipping')
                    # grads_and_vars = optimizer.compute_gradients(self.cost, var_list=non_encoder_params)
                    # clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
                    # train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)
                train_op = optimizer.minimize(self.cost, global_step=global_step, var_list=non_encoder_params)

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
            dev_loss_writer = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "loss"),
            )
            dev_eval_writer1 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Rat5"),
            )
            dev_eval_writer2 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Rat10"),
            )
            dev_eval_writer3 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Pat5"),
            )
            dev_eval_writer4 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "Pat10"),
            )
            dev_eval_writer5 = tf.summary.FileWriter(
                os.path.join(self.args.save_dir, "summaries", "dev", "MAP"),
            )

            loss = tf.placeholder(tf.float32)
            loss_summary = tf.summary.scalar("loss", loss)
            dev_eval = tf.placeholder(tf.float32)
            dev_summary = tf.summary.scalar("TP_evaluation", dev_eval)
            cost = tf.placeholder(tf.float32)
            cost_summary = tf.summary.scalar("cost", cost)

            if self.args.save_dir != "":
                checkpoint_dir = os.path.join(self.args.save_dir, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

            patience = 5 if 'patience' not in self.args else self.args.patience
            unchanged = 0
            max_epoch = self.args.max_epoch
            for epoch in xrange(max_epoch):
                unchanged += 1
                if unchanged > patience:
                    break

                train_batches = list(myio.create_batches(
                    df, ids_corpus, 'train', self.args.batch_size, self.padding_id, N_neq=self.args.n_neg)
                )
                N = len(train_batches)
                train_loss = 0.0
                train_cost = 0.0

                i = 0
                for batch in train_batches:
                    i += 1
                    titles_b, bodies_b, tag_labels_b, tuples_b = batch

                    if i % 10 == 0 and self.args.testing:
                        print 'labels in batch: ', np.sum(np.sum(tag_labels_b, 0) > 0)
                        emb = sess.run(self.embeddings)
                        print '\nemb {}\n'.format(emb[10][0:10])

                    cur_step, cur_loss, cur_cost = self.train_batch(
                        titles_b, bodies_b, tag_labels_b, tuples_b,
                        train_op, global_step, sess
                    )

                    loss_sum, cost_sum = sess.run([loss_summary, cost_summary], {loss: cur_loss, cost: cur_cost})
                    train_loss_writer.add_summary(loss_sum, cur_step)
                    train_loss_writer.flush()
                    train_cost_writer.add_summary(cost_sum, cur_step)
                    train_cost_writer.flush()

                    train_loss += cur_loss
                    train_cost += cur_cost

                    if i % 10 == 0:
                        myio.say("\r{}/{}".format(i, N))

                    if i == N-1 or (i % 10 == 0 and self.args.testing):  # EVAL
                        dev_loss = 0

                        if dev:
                            dev_loss, (dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_MAP) = self.evaluate(dev, sess)

                            summary = sess.run(loss_summary, {loss: dev_loss})
                            dev_loss_writer.add_summary(summary, cur_step)
                            dev_loss_writer.flush()

                            summary = sess.run(dev_summary, {dev_eval: dev_RAT5})
                            dev_eval_writer1.add_summary(summary, cur_step)
                            dev_eval_writer1.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_RAT10})
                            dev_eval_writer2.add_summary(summary, cur_step)
                            dev_eval_writer2.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_PAT5})
                            dev_eval_writer3.add_summary(summary, cur_step)
                            dev_eval_writer3.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_PAT10})
                            dev_eval_writer4.add_summary(summary, cur_step)
                            dev_eval_writer4.flush()
                            summary = sess.run(dev_summary, {dev_eval: dev_MAP})
                            dev_eval_writer5.add_summary(summary, cur_step)
                            dev_eval_writer5.flush()

                            feed_dict = {}
                            for param_name, param_norm in self.get_pnorm_stat(sess).iteritems():
                                feed_dict[p_norm_placeholders[param_name]] = param_norm
                            _p_norm_sum = sess.run(p_norm_summary_op, feed_dict)
                            p_norm_summary_writer.add_summary(_p_norm_sum, cur_step)

                        if test:
                            test_loss, (test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_MAP) = self.evaluate(test, sess)

                        if dev_MAP > best_dev_performance:
                            unchanged = 0
                            best_dev_performance = dev_MAP
                            result_table.add_row(
                                [epoch, cur_step, dev_PAT5, dev_PAT10, dev_RAT5, dev_RAT10, dev_MAP,
                                 test_PAT5, test_PAT10, test_RAT5, test_RAT10, test_MAP])
                            if self.args.save_dir != "":
                                self.save(sess, checkpoint_prefix, cur_step)

                        myio.say("\r\n\nEpoch {}\tcost={:.3f}\tloss={:.3f}\tDevLoss={:.3f}\tMAP={:.2f},{:.2f}\n".format(
                            epoch,
                            train_cost / (i+1),  # i.e. divided by N training batches
                            train_loss / (i+1),  # i.e. divided by N training batches
                            dev_loss,
                            dev_MAP,
                            best_dev_performance
                        ))
                        myio.say("\n{}\n".format(result_table))
                        myio.say("\tp_norm: {}\n".format(
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


class LstmMultiTagsClassifier(ModelMultiTagsClassifier):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        self.encoder_params = []
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


class BiRNNMultiTagsClassifier(ModelMultiTagsClassifier):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        self.encoder_params = []
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


class CnnMultiTagsClassifier(ModelMultiTagsClassifier):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        self.encoder_params = []
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
                            # padding="SAME",
                            name="conv-titles")
                        # self.conv_t = conv_t  # batch_size x seq_len i.e. time_steps x 1 x hidden_dim

                        # Apply nonlinearity
                        nl_fun = get_activation_by_name(self.args.activation)
                        h_t = nl_fun(tf.nn.bias_add(conv_t, b), name="act-titles")
                        # self.h_t = h_t  # batch_size x seq_len i.e. time_steps x 1 x hidden_dim

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
                        # self.pooled_t = pooled_t  # batch_size x 1 x 1 x hidden_dim

                        pooled_outputs_t.append(pooled_t)

                    with tf.name_scope('bodies_output'):
                        conv_b = tf.nn.conv2d(
                            self.embedded_bodies_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            # padding="SAME",
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
            with tf.name_scope('encodings'):
                # batch * d
                h_final = (self.t_state + self.b_state) * 0.5
                h_final = tf.nn.dropout(h_final, 1.0 - self.dropout_prob)
                self.h_final = self.normalize_2d(h_final)


class GruMultiTagsClassifier(ModelMultiTagsClassifier):

    def __init__(self, args, embedding_layer, output_dim, weights=None):
        self.args = args
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.output_dim = output_dim
        self.params = {}
        self.encoder_params = []
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
            # current_state = last state of every layer in the network as an LSTMStateTuple

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
            # current_state = last state of every layer in the network as an LSTMStateTuple

            if self.args.normalize:
                self.b_states_series = self.normalize_3d(self.b_states_series)

            if self.args.average == 1:
                self.b_state = self.average_without_padding(self.b_states_series, self.bodies_words_ids_placeholder)
            elif self.args.average == 0:
                self.b_state = self.b_current_state[0]
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
