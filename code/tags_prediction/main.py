import argparse
import os
import pickle
import sys
import time

import myio
from collections import Counter
from nn import EmbeddingLayer
from utils import load_embedding_iterator
from utils.statistics import read_df


# from bilstm_model import Model
from model import Model


def baselines_eval(train_data, dev_data, test_data):
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support

    counts_per_class = np.zeros(100)

    training_data = 0

    for train_b in train_data:
        tags = np.array(train_b[2]).astype(np.float32)
        training_data += tags.shape[0]
        count_per_class = tags.sum(axis=0)
        assert len(count_per_class) == 100
        counts_per_class += count_per_class

    assert training_data == 159975

    majority_per_class = (counts_per_class > 159975/2).astype(np.int32)
    prob_per_class = np.round(counts_per_class / 159975.)

    target = []
    for dev_b in dev_data:
        target.extend(dev_b[2])
    target = np.array(target).astype(np.float32)
    majority_predict = np.tile(majority_per_class, (target.shape[0], 1))
    prob_predict = np.tile(prob_per_class, (target.shape[0], 1))
    assert target.shape == majority_predict.shape == prob_predict.shape

    print '\nDEV MACRO EVAL ON MAJORITY PREDICTION\n', precision_recall_fscore_support(target, majority_predict, average='macro')
    print '\nDEV MICRO EVAL ON MAJORITY PREDICTION\n', precision_recall_fscore_support(target, majority_predict, average='micro')
    print '\nDEV MACRO EVAL ON PROBABILITY PREDICTION\n', precision_recall_fscore_support(target, prob_predict, average='macro')
    print '\nDEV MICRO EVAL ON PROBABILITY PREDICTION\n', precision_recall_fscore_support(target, prob_predict, average='micro')

    target = []
    for test_b in test_data:
        target.extend(test_b[2])
    target = np.array(target).astype(np.float32)
    majority_predict = np.tile(majority_per_class, (target.shape[0], 1))
    prob_predict = np.tile(prob_per_class, (target.shape[0], 1))
    assert target.shape == majority_predict.shape == prob_predict.shape

    print '\nTEST MACRO EVAL ON MAJORITY PREDICTION\n', precision_recall_fscore_support(target, majority_predict, average='macro')
    print '\nTEST MICRO EVAL ON MAJORITY PREDICTION\n', precision_recall_fscore_support(target, majority_predict, average='micro')
    print '\nTEST MACRO EVAL ON PROBABILITY PREDICTION\n', precision_recall_fscore_support(target, prob_predict, average='macro')
    print '\nTEST MICRO EVAL ON PROBABILITY PREDICTION\n', precision_recall_fscore_support(target, prob_predict, average='micro')


def create_embedding_layer(raw_corpus, n_d, embs=None, cut_off=2,
                           unk="<unk>", padding="<padding>", fix_init_embs=True):

    cnt = Counter(w for id, pair in raw_corpus.iteritems() for x in pair for w in x)
    cnt[unk] = cut_off + 1
    cnt[padding] = cut_off + 1
    embedding_layer = EmbeddingLayer(
            n_d=n_d,
            # vocab=(w for w,c in cnt.iteritems() if c > cut_off),
            vocab=[unk, padding],
            embs=embs,
            fix_init_embs=fix_init_embs
        )
    # print embedding_layer.oov_id
    # print embedding_layer.vocab_map.keys()[0:5]
    # print embedding_layer.embeddings
    # print embedding_layer.embeddings_trainable
    return embedding_layer


def main():
    df = read_df(args.df_path)
    df = df.fillna(u'')

    label_tags = pickle.load(open(args.tags_file, 'rb'))

    raw_corpus = myio.read_corpus(args.corpus, with_tags=True)
    embedding_layer = create_embedding_layer(
                raw_corpus,
                n_d=240,
                cut_off=1,
                embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
            )

    ids_corpus_tags = myio.make_tag_labels(df, label_tags)

    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, ids_corpus_tags, max_len=args.max_seq_len)

    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        weights = myio.create_idf_weights(args.corpus, embedding_layer, with_tags=True)

    dev = myio.create_batches(df, ids_corpus, 'dev', args.batch_size, padding_id, pad_left=not args.average)
    test = myio.create_batches(df, ids_corpus, 'test', args.batch_size, padding_id, pad_left=not args.average)
    train = myio.create_batches(df, ids_corpus, 'train', args.batch_size, padding_id, pad_left=not args.average)
    print '{} batches of {} instances in dev, {} in test and {} in train.'.format(
        len(dev), args.batch_size, len(test), len(train))

    # baselines_eval(train, dev, test)

    model = Model(args, embedding_layer, len(label_tags), weights=weights if args.reweight else None)
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
    main()

