import tensorflow as tf
from initialization import random_init
import numpy as np


class EmbeddingLayer(object):
    """
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)

        Inputs
        ------

        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs

    """
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        if embs is not None:
            lst_words = []
            vocab_map = {}  # i.e. {'word1' :1, 'word2': 2 ...}
            emb_vals = []
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                print("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            print("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)  # continue adding words in the embedding matrix
                    # out-of-vocab token is initialized as zero vector
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals).astype(np.float32)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:  # no embeddings given
            lst_words = [ ]
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))  # random initialization of whole embedding matrix
            self.init_end = -1

        if oov is not None and oov is not False:  # if oov is given, it should be already in vocab_map
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:  # if oov is not given then we set the id to -1 so that it's not used
            self.oov_tok = None
            self.oov_id = -1

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

        self._initialize_params(emb_vals, True)
        self._initialize_ops()

    def _initialize_params(self, init_embeddings, trainable):
        # THIS I THE EMBEDDING LAYER, which takes as input the sequence of word ids
        # and gives as output the embeddings of words in each sentence
        with tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.n_V, self.n_d], -1.0, 1.0),
                trainable=trainable,
                name="W")
            if init_embeddings is not None:
                self.embeddings = tf.assign(self.embeddings, init_embeddings)
            self.embeddings_trainable = self.embeddings

    def _initialize_ops(self):
        self.x = tf.placeholder(tf.int32, [None], name="input_x")
        self._forward = tf.nn.embedding_lookup(self.embeddings, self.x)

    def forward(self, x):
        res = tf.Session().run(self._forward, {self.x: x})
        return res

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [lst_words[i] if i < n_V else "<err>" for i in ids]

    def map_to_ids(self, words, filter_oov=False):
        """
            map the list of string tokens into a numpy array of integer IDs

            Inputs
            ------

            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array


            Outputs
            -------

            return the numpy array of word IDs

        """
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            def not_oov(xx):
                return xx != oov_id
            return np.array(
                # filter will only keep the non oov_id's
                filter(not_oov, [vocab_map.get(x, oov_id) for x in words]),
                dtype="int32"
                )
        else:
            return np.array(
                    [vocab_map.get(x, oov_id) for x in words],
                    dtype="int32"
                )


class Layer(object):
    """
        Basic neural layer -- y = f(Wx+b)
        forward(x) returns y

        Inputs
        ------

        n_in            : input dimension
        n_out           : output dimension
        activation      : the non-linear activation function to apply
        has_bias        : whether to include the bias term b in the computation

    """

    def __init__(self, n_in, n_out, activation, has_bias=True, name="Layer"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.has_bias = has_bias
        self.name = name

        with tf.name_scope(name):
            self.initialize_placeholders()
            self.initialize_params(n_in, n_out)
            self.initialize_ops()

    def initialize_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in], name="input_x")
        self.y = tf.placeholder(tf.float32, [None, self.n_out], name="output_y")

    def initialize_params(self, n_in, n_out):
        self._initialize_params(n_in, n_out)

    def _initialize_params(self, n_in, n_out):
        W_vals = random_init((n_in, n_out))
        if self.activation == tf.nn.softmax:
            W_vals *= 0.001
        if self.activation == tf.nn.relu:
            b_vals = np.ones(n_out, dtype=np.float32) * 0.01
        else:
            b_vals = random_init((n_out,))
        self.W = tf.Variable(W_vals, name="W", dtype=tf.float32)
        if self.has_bias:
            self.b = tf.Variable(b_vals, name="b", dtype=tf.float32)

    def initialize_ops(self):
        with tf.name_scope("output"):
            if self.has_bias:
                self.scores = tf.nn.xw_plus_b(self.x, self.W, self.b, name="scores")
            else:
                self.scores = tf.multiply(self.x, self.W)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.y
            )
            self.loss = tf.reduce_mean(losses)

    @property
    def params(self):
        if self.has_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    @params.setter
    def params(self, param_list):
        self.W = tf.assign(self.W, param_list[0])
        if self.has_bias:
            self.b = tf.assign(self.b, param_list[1])


if __name__ == '__main__':

    def test_embedding_layer():
        from qr import myio
        from utils import load_embedding_iterator
        raw_corpus = myio.read_corpus('/home/christina/Documents/Thesis/taolei_code/askubuntu/texts_raw_fixed.txt')
        embedding_layer = myio.create_embedding_layer(
                    raw_corpus,
                    n_d=200,
                    cut_off=1,
                    embs=load_embedding_iterator(
                        '/home/christina/Documents/Thesis/taolei_code/askubuntu/vector/vectors_pruned.200.txt.gz'
                    )
                )
        ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=100)
        print("vocab size={}, corpus size={}\n".format(
                embedding_layer.n_V,
                len(raw_corpus)
            ))
        padding_id = embedding_layer.vocab_map["<padding>"]

        # for i, item in enumerate(ids_corpus.iteritems()):
        #     if i < 1:
        #         print item[0]
        #         print item[1]
        train = myio.read_annotations('/home/christina/Documents/Thesis/taolei_code/askubuntu/train_random.txt')
        train_batches = myio.create_batches(
            ids_corpus, train, 40, padding_id, pad_left=1
        )
        print("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel()) + len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))

        idts, idbs, idps = train_batches[0]
        # print idts
        # print idbs
        # print idps

        with tf.Session() as sess:
            print idts.ravel(), '\n\n'
            res = embedding_layer.forward(idts.ravel())
            print 'res ', type(res)

    test_embedding_layer()


    def train_layer_mnist():
        import gzip
        import pickle
        from initialization import get_activation_by_name
        with gzip.open('/home/christina/Documents/Python-Sublime/mnist.pkl.gz', 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
                # print train_set[0].shape, train_set[1].shape  # (50000, 784) (50000,)
        # print set(train_set[1])  # {0,1,2,3,4,5,6,7,8,9}
        train_x = train_set[0]
        train_y = train_set[1]

        def one_hot(targets, nb_classes):
            one_hot_targets = np.eye(nb_classes)[targets]
            return one_hot_targets

        train_y = one_hot(train_y, 10)

        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():

                a = get_activation_by_name('relu')
                model = Layer(784, 10, a, has_bias=True)

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(model.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                def train_step(x_batch, y_batch):
                    feed_dict = {
                        model.x: x_batch,
                        model.y: y_batch
                    }
                    _, step, loss = sess.run(
                        [train_op, global_step, model.loss],
                        feed_dict
                    )
                    print("step {}, loss {:g}".format(step, loss))

            sess.run(tf.global_variables_initializer())
            # print tf.trainable_variables()
            # print tf.global_variables()
            # print [n.name for n in tf.get_default_graph().as_graph_def().node]

            for i in range(10):
                train_images, train_labels = train_x[i*10: (i+1)*10], train_y[i*10: (i+1)*10]
                train_step(train_images, train_labels)
