import argparse
from qa import myio as qaio
import sys
import time
import os
import pickle
from tags_prediction import myio as tpio
from tags_prediction.statistics import read_df
from utils import load_embedding_iterator
import myio


def main():
    raw_corpus = qaio.read_corpus(args.corpus)
    embedding_layer = qaio.create_embedding_layer(
        raw_corpus,
        n_d=args.hidden_dim,
        cut_off=args.cut_off,
        embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
    )
    print("vocab size={}, corpus size={}\n".format(embedding_layer.n_V, len(raw_corpus)))
    if args.reweight:
        weights = qaio.create_idf_weights(args.corpus, embedding_layer)

    label_tags = pickle.load(open(args.tags_file, 'rb'))
    if isinstance(label_tags, dict):
        print 'from dict labels to list.'
        label_tags = label_tags.keys()
    print '\nloaded {} tags'.format(len(label_tags))

    raw_corpus_tags = tpio.read_corpus(args.corpus_w_tags, with_tags=True)
    ids_corpus_tags = tpio.map_corpus(raw_corpus_tags, embedding_layer, label_tags, max_len=args.max_seq_len)

    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.dev:
        dev = qaio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus_tags, dev, padding_id)

    if args.test:
        test = qaio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus_tags, test, padding_id)

    if args.train:
        start_time = time.time()
        train = qaio.read_annotations(args.train)
        train_batches = myio.create_batches(ids_corpus_tags, train, args.batch_size, padding_id)
        print("{} to create batches\n".format(time.time()-start_time))
        print("{} batches, {} tokens in total, {} triples in total\n".format(
            len(train_batches),
            sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
            sum(len(x[2].ravel()) for x in train_batches)
        ))

        if args.layer.lower() == "lstm":
            from models import LstmQRTP as Model
        elif args.layer.lower() == "bilstm":
            from models import BiLstmQRTP as Model
        elif args.layer.lower() == "cnn":
            from models import CnnQRTP as Model
        elif args.layer.lower() == "gru":
            from models import GruQRTP as Model

        model = Model(args, embedding_layer, len(label_tags), weights=weights if args.reweight else None)
        model.ready()
        print 'total params: ', model.num_parameters()

        assert not (args.load_pre_trained_part != "" and args.load_trained_vars != "")
        if args.load_trained_vars:
            assign_ops = model.load_trained_vars(args.load_trained_vars)
        elif args.load_pre_trained_part:
            assign_ops = model.load_pre_trained_part(args.load_pre_trained_part)
        else:
            assign_ops = None

        model.train_model(
            train_batches, dev=dev if args.dev else None, test=test if args.test else None, assign_ops=assign_ops
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--corpus_w_tags", type=str)
    argparser.add_argument("--tags_file", type=str)

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
    # argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--reweight", type=int, default=1)
    argparser.add_argument("--layer", type=str, default="lstm")
    argparser.add_argument("--concat", type=int, default=0)

    argparser.add_argument("--threshold", type=float, default=0.5)
    argparser.add_argument("--performance", type=str, default="dev_mrr")
    argparser.add_argument("--qr_weight", type=float, default=1.)
    argparser.add_argument("--tp_weight", type=float, default=1.)

    argparser.add_argument("--load_trained_vars", type=str, default="")
    argparser.add_argument("--load_pre_trained_part", type=str, default="")

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print args
    print ""
    main()

