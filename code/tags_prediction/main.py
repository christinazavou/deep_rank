import argparse
import os
import pickle
import sys
import time

import myio
from tags_prediction.statistics import read_df
from utils import load_embedding_iterator, create_embedding_layer


def main():
    s_time = time.time()
    df = read_df(args.df_path)
    df = df.fillna(u'')

    label_tags = pickle.load(open(args.tags_file, 'rb'))
    print '\nloaded {} tags'.format(len(label_tags))

    raw_corpus = myio.read_corpus(args.corpus_w_tags, with_tags=True)

    embedding_layer = create_embedding_layer(
        n_d=240,
        embs=load_embedding_iterator(args.embeddings),
        only_words=False if args.use_embeddings else True,
        # only_words will take the words from embedding file and make random initial embeddings
        trainable=args.trainable
    )

    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, label_tags, max_len=args.max_seq_len)

    print("vocab size={}, corpus size={}\n".format(embedding_layer.n_V, len(raw_corpus)))

    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        weights = myio.create_idf_weights(args.corpus_w_tags, embedding_layer, with_tags=True)

    if args.layer.lower() == "lstm":
        from models import LstmMultiTagsClassifier as Model
    elif args.layer.lower() in ["bilstm", "bigru"]:
        from models import BiRNNMultiTagsClassifier as Model
    elif args.layer.lower() == "cnn":
        from models import CnnMultiTagsClassifier as Model
    elif args.layer.lower() == "gru":
        from models import GruMultiTagsClassifier as Model
    else:
        raise Exception("no correct layer given")

    if args.cross_val:
        train, dev, test = myio.create_cross_val_batches(df, ids_corpus, args.batch_size, padding_id)
    else:
        dev = myio.create_batches(df, ids_corpus, 'dev', args.batch_size, padding_id)
        test = myio.create_batches(df, ids_corpus, 'test', args.batch_size, padding_id)
        train = myio.create_batches(df, ids_corpus, 'train', args.batch_size, padding_id)

    print '{} batches of {} instances in dev, {} in test and {} in train.'.format(
        len(dev), args.batch_size, len(test), len(train))
    print time.time() - s_time

    # baselines_eval(train, dev, test)

    model = Model(args, embedding_layer, len(label_tags), weights=weights if args.reweight else None)
    model.ready()

    print 'total params: ', model.num_parameters()

    if args.load_pre_trained_part:
        # need to remove the old assigns to embeddings
        model.init_assign_ops = model.load_pre_trained_part(args.load_pre_trained_part)
    print '\nmodel init_assign_ops: {}\n'.format(model.init_assign_ops)

    model.train_model(train, dev=dev, test=test)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus_w_tags", type=str)
    argparser.add_argument("--df_path", type=str)
    argparser.add_argument("--tags_file", type=str)
    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--load_pre_trained_part", type=str, default="")
    argparser.add_argument("--testing", type=int, default=0)

    argparser.add_argument("--use_embeddings", type=int, default=1)
    argparser.add_argument("--trainable", type=int, default=1)
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)

    argparser.add_argument("--batch_size", type=int, default=128)
    argparser.add_argument("--cross_val", type=int, default=0)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--optimizer", type=str, default="adam")
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--reweight", type=int, default=1)
    argparser.add_argument("--normalize", type=int, default=1)
    # average 1 = mean pooling for any layer
    # average 0 = last pooling for RNNs max pooling for CNN
    # average 2 = max pooling for RNNs
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--layer", type=str, default="lstm")
    argparser.add_argument("--concat", type=int, default=0)
    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--performance", type=str, default="R@10")  # P@5, R@10
    argparser.add_argument("--loss", type=str, default="mean")  # sum, max
    argparser.add_argument("--entropy", type=int, default=1)

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print '\n{}\n'.format(args)
    main()

