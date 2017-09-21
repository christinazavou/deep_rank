import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell


def bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs_embedded,
        input_lengths,
        scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs_embedded,
            sequence_length=input_lengths,
            dtype=tf.float32,
            swap_memory=True,
            scope=scope
        )
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat((fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat((fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1, name='bidirectional_concat')
                return state
            elif isinstance(fw_state, tuple) and isinstance(bw_state, tuple) and (len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state))
                return state
            else:
                raise ValueError('unknown state type: {}'.format((fw_state, bw_state)))

        state = concatenate_state(fw_state, bw_state)

        return outputs, state


class HANClassifierModel(object):

    def __init__(self, args, embedding_layer, word_cell, sent_cell,  word_hid_dim, sent_hid_dim, word_attention_size,
                 sent_attention_size, word_sequence_length, sent_sequence_length, scope=None):

        self.args = args
        self.embeddings = embedding_layer.embeddings
        self.embedding_size = embedding_layer.n_d

        self.word_cell = word_cell
        self.word_attention_size = word_attention_size
        self.word_hid_dim = word_hid_dim
        self.word_sequence_length = word_sequence_length

        self.sent_cell = sent_cell
        self.sent_attention_size = sent_attention_size
        self.sent_hid_dim = sent_hid_dim
        self.sent_sequence_length = sent_sequence_length

        with tf.variable_scope(scope or 'tcm') as scope:

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')  # UN-PADDED

            # [document]
            self.sent_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')  # UN-PADDED

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')

            self.dropout_keep_proba = tf.placeholder(tf.float32, name="dropout_keep_proba")

            self._init_embedding(scope)

            self._init_body(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            self.loss = tf.reduce_mean(self.cross_entropy)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))

            self.tvars = tf.trainable_variables()

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding"):
                self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):

            """ -------------------------------------------WORD LEVEL----------------------------------------------- """

            word_level_inputs = tf.reshape(
                self.inputs_embedded,
                [-1, self.word_sequence_length, self.embedding_size]
            )
            self.word_level_inputs = word_level_inputs

            word_level_lengths = tf.reshape(self.word_lengths, [-1])

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell,
                    self.word_cell,
                    word_level_inputs,
                    word_level_lengths,
                    scope=scope
                )
                self.word_hid_dim = self.word_hid_dim * 2  # DUE TO BIDIRECTIONAL

                # word_encoder_output, _ = tf.nn.dynamic_rnn(
                #     self.word_cell,
                #     word_level_inputs,
                #     word_level_lengths,
                #     dtype=tf.float32
                # )

                self.word_encoder_output = word_encoder_output

                with tf.variable_scope('attention'):

                    w_omega_w = tf.Variable(tf.random_normal([self.word_hid_dim, self.word_attention_size], stddev=0.1))
                    b_omega_w = tf.Variable(tf.random_normal([self.word_attention_size], stddev=0.1))
                    u_omega_w = tf.Variable(tf.random_normal([self.word_attention_size], stddev=0.1))

                    v_w = tf.tanh(
                        tf.matmul(tf.reshape(word_encoder_output, [-1, self.word_hid_dim]), w_omega_w)
                        +
                        tf.reshape(b_omega_w, [1, -1])
                    )
                    vu_w = tf.matmul(v_w, tf.reshape(u_omega_w, [-1, 1]))
                    exps_w = tf.reshape(tf.exp(vu_w), [-1, self.word_sequence_length])

                    alphas_w = exps_w / tf.reshape(tf.reduce_sum(exps_w, 1), [-1, 1])

                    # Output of previous layer (i.e. the encodings given here) reduced with the attention vector
                    word_level_output = tf.reduce_sum(
                        word_encoder_output * tf.reshape(alphas_w, [-1, self.word_sequence_length, 1]),
                        1
                    )
                    self.word_level_output = word_level_output

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output,
                        keep_prob=self.dropout_keep_proba,
                    )

            """ ---------------------------------------SENTENCE LEVEL----------------------------------------------- """

            sent_level_inputs = tf.reshape(
                word_level_output,
                [-1, self.sent_sequence_length, self.word_hid_dim]
            )
            self.sent_level_inputs = sent_level_inputs

            with tf.variable_scope('sentence') as scope:
                sent_encoder_output, _ = bidirectional_rnn(
                    self.sent_cell,
                    self.sent_cell,
                    sent_level_inputs,
                    self.sent_lengths,
                    scope=scope
                )
                self.sent_hid_dim = self.sent_hid_dim * 2  # DUE TO BIDIRECTIONAL

                self.sent_encoder_output = sent_encoder_output

                with tf.variable_scope('attention'):

                    w_omega_s = tf.Variable(tf.random_normal([self.sent_hid_dim, self.sent_attention_size], stddev=0.1))
                    b_omega_s = tf.Variable(tf.random_normal([self.sent_attention_size], stddev=0.1))
                    u_omega_s = tf.Variable(tf.random_normal([self.sent_attention_size], stddev=0.1))

                    v_s = tf.tanh(
                        tf.matmul(tf.reshape(sent_encoder_output, [-1, self.sent_hid_dim]), w_omega_s)
                        +
                        tf.reshape(b_omega_s, [1, -1])
                    )
                    vu_s = tf.matmul(v_s, tf.reshape(u_omega_s, [-1, 1]))
                    exps_s = tf.reshape(tf.exp(vu_s), [-1, self.sent_sequence_length])

                    alphas_s = exps_s / tf.reshape(tf.reduce_sum(exps_s, 1), [-1, 1])

                    # Output of previous layer (i.e. the encodings given here) reduced with the attention vector
                    sent_level_output = tf.reduce_sum(
                        sent_encoder_output * tf.reshape(alphas_s, [-1, self.sent_sequence_length, 1]),
                        1
                    )
                    self.sent_level_output = sent_level_output

                with tf.variable_scope('dropout'):
                    sent_level_output = layers.dropout(
                        sent_level_output,
                        keep_prob=self.dropout_keep_proba,
                    )

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sent_level_output, self.classes, activation_fn=None
                )
                self.prediction = tf.argmax(self.logits, axis=-1)


if __name__ == '__main__':

    with tf.Session() as session:

        model = HANClassifierModel(
            vocab_size=20,
            embedding_size=4,
            classes=2,

            # word_cell=GRUCell(6),
            word_cell=LSTMCell(6),
            # sent_cell=GRUCell(8),
            sent_cell=LSTMCell(8),

            word_attention_size=7,
            sent_attention_size=9,

            word_hid_dim=6,
            sent_hid_dim=8,
            word_sequence_length=4,
            sent_sequence_length=3,
        )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        session.run(tf.global_variables_initializer())

        fd = {
            model.dropout_keep_proba: 0.5,
            model.inputs:
                [
                    [
                        [5, 4, 1, 0],
                        [3, 3, 6, 7],
                        [6, 7, 0, 0]
                    ],
                    [
                        [2, 2, 1, 0],
                        [3, 3, 6, 7],
                        [0, 0, 0, 0]
                    ]
                ],
            model.word_lengths:
                [
                    [3, 4, 2],
                    [3, 4, 0],
                ],
            model.sent_lengths: [3, 2],
            model.labels: [0, 1],
        }

        logits, pred = session.run(
            [model.logits, model.prediction],
            fd
        )
        print('logits: ', logits, ' pred: ', pred)
        session.run(train_op, fd)
