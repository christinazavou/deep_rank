import argparse
import os
import sys
import time

from qa.WideNDeep.main_model import Model
from qa.WideNDeep.myio import read_eval_annotations, read_train_annotations, create_eval_batches, \
    create_train_batches
from qa.myio import create_embedding_layer, map_corpus
from tags_prediction.myio import read_corpus
from utils import load_embedding_iterator


def main():
    raw_corpus = read_corpus(args.corpus, with_tags=True, test=-1)
    embedding_layer = create_embedding_layer(
                raw_corpus,
                n_d=args.hidden_dim,
                cut_off=args.cut_off,
                embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
            )
    ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    dev = read_eval_annotations(args.dev)
    dev = create_eval_batches(ids_corpus, dev, padding_id, pad_left=not args.average)

    test = read_eval_annotations(args.test)
    test = create_eval_batches(ids_corpus, test, padding_id, pad_left=not args.average)

    train = read_train_annotations(args.train)
    train_batches = create_train_batches(
        ids_corpus, train, args.batch_size, padding_id, pad_left=not args.average
    )

    model = Model(args, embedding_layer, 42)
    model.ready()

    assert not (args.load_pre_trained_part != "" and args.load_trained_vars != "")
    if args.load_trained_vars:
        assign_ops = model.load_trained_vars(args.load_trained_vars)
    elif args.load_pre_trained_part:
        assign_ops = model.load_pre_trained_part(args.load_pre_trained_part)
    else:
        assign_ops = None

    model.train_model(
        train_batches,
        dev=dev if args.dev else None,
        test=test if args.test else None,
        assign_ops=assign_ops
    )


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
    argparser.add_argument("--dnn_lr", type=float, default=1e-3)
    argparser.add_argument("--linear_lr", type=float, default=0.2)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--normalize", type=int, default=1)
    # argparser.add_argument("--reweight", type=int, default=1)

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
