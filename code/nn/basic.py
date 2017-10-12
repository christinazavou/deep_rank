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
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True, trainable=True):

        self.init_embeddings = None
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
            lst_words = []
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

        self._initialize_params(emb_vals, trainable)

    def _initialize_params(self, init_embeddings, trainable):
        # THIS I THE EMBEDDING LAYER, which takes as input the sequence of word ids
        # and gives as output the embeddings of words in each sentence
        with tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([self.n_V, self.n_d], -1.0, 1.0),
                trainable=trainable,
                name="W")
            if init_embeddings is not None:
                self.init_embeddings = init_embeddings

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
