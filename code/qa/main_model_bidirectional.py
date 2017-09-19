import tensorflow as tf
from nn import get_activation_by_name


from main_model import Model as BasicModel


class Model(BasicModel):

    def __init__(self, args, embedding_layer, weights=None):
        self.args = args
        if args is not None and args.average is 0:  # i.e. concatenation of states will be used
            assert self.args.hidden_dim % 2 == 0, ' can not concat. either use average 1 or an even hidden dimension '
        self.embedding_layer = embedding_layer
        self.embeddings = embedding_layer.embeddings
        self.padding_id = embedding_layer.vocab_map["<padding>"]
        self.weights = weights
        self.params = {}

    def _initialize_graph(self):

        with tf.name_scope('input'):
            self.titles_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='titles_ids')
            self.bodies_words_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')
            self.pairs_ids_placeholder = tf.placeholder(tf.int32, [None, None], name='bodies_ids')  # LENGTH = 3 OR 22

            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

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
            # output_fw = a Tensor shaped: [batch_size, max_time, cell_fw.output_size]
            # output_bw = a Tensor shaped: [batch_size, max_time, cell_bw.output_size].
            # output_states: A tuple (output_state_fw, output_state_bw)

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
            # output_fw = a Tensor shaped: [batch_size, max_time, cell_fw.output_size]
            # output_bw = a Tensor shaped: [batch_size, max_time, cell_bw.output_size].
            # output_states: A tuple (output_state_fw, output_state_bw)

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
            # batch * d
            h_final = (self.t_state_vec + self.b_state_vec) * 0.5
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

    def _find_sequence_length(self, ids):
        s = tf.not_equal(ids, self.padding_id)
        s = tf.cast(s, tf.int32)
        s = tf.reduce_sum(s, axis=1)
        return s
