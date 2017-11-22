import argparse
from qr import myio as qaio
import sys
import time
import os
import pickle
from tags_prediction import myio as tpio
from utils import load_embedding_iterator, create_embedding_layer
import myio
from datetime import datetime


def main():
    print 'Starting at: {}\n'.format(datetime.now())
    raw_corpus = qaio.read_corpus(args.corpus)
    embedding_layer = create_embedding_layer(
        n_d=200,
        embs=load_embedding_iterator(args.embeddings),
        only_words=False if args.use_embeddings else True,
        trainable=args.trainable
    )
    print("vocab size={}, corpus size={}\n".format(embedding_layer.n_V, len(raw_corpus)))
    if args.reweight:
        weights = qaio.create_idf_weights(args.corpus, embedding_layer)

    label_tags = pickle.load(open(args.tags_file, 'rb'))
    print '\nloaded {} tags'.format(len(label_tags))

    raw_corpus_tags = tpio.read_corpus(args.corpus_w_tags, with_tags=True)
    ids_corpus_tags = tpio.map_corpus(raw_corpus_tags, embedding_layer, label_tags, max_len=args.max_seq_len)

    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.dev:
        dev = qaio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus_tags, dev, padding_id, N_neg=args.n_neg, samples_file=args.samples_file)

    if args.test:
        test = qaio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus_tags, test, padding_id, N_neg=args.n_neg, samples_file=args.samples_file)

    if args.train:
        train = qaio.read_annotations(args.train)

        if args.layer.lower() == "lstm":
            from models import LstmQRTP as Model
        elif args.layer.lower() in ["bilstm", "bigru"]:
            from models import BiRNNQRTP as Model
        elif args.layer.lower() == "cnn":
            from models import CnnQRTP as Model
        elif args.layer.lower() == "gru":
            from models import GruQRTP as Model
        else:
            raise Exception("no correct layer given")

        model = Model(args, embedding_layer, len(label_tags), weights=weights if args.reweight else None)
        model.ready()
        print 'total (non) trainable params: ', model.num_parameters()

        if args.load_pre_trained_part:
            # need to remove the old assigns to embeddings
            model.init_assign_ops = model.load_pre_trained_part(args.load_pre_trained_part)
        print '\nmodel init_assign_ops: {}\n'.format(model.init_assign_ops)

        model.train_model(
            ids_corpus_tags, train, dev=dev if args.dev else None, test=test if args.test else None
        )
    print '\nEnded at: {}'.format(datetime.now())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--corpus_w_tags", type=str)
    argparser.add_argument("--tags_file", type=str)
    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--load_pre_trained_part", type=str, default="")
    argparser.add_argument("--testing", type=int, default=0)

    argparser.add_argument("--use_embeddings", type=int, default=1)  # refers to word embeddings
    argparser.add_argument("--trainable", type=int, default=1)
    argparser.add_argument("--load_only_embeddings", type=int, default=0)  # refers to word embeddings

    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)
    # average 1 = mean pooling for any layer
    # average 0 = last pooling for RNNs max pooling for CNN
    # average 2 = max pooling for RNNs
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--optimizer", type=str, default="adam")
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

    argparser.add_argument("--mlp_dim_tp", type=int, default=0)

    argparser.add_argument("--loss_qr", type=str, default="loss0")
    argparser.add_argument("--entropy_qr", type=int, default=0)
    argparser.add_argument("--loss_tp", type=str, default="loss2")
    argparser.add_argument("--entropy_tp", type=int, default=1)

    argparser.add_argument("--performance", type=str, default="")  # dev_map_qr or dev_map_tp
    argparser.add_argument("--qr_weight", type=float, default=1.)
    argparser.add_argument("--tp_weight", type=float, default=1.)
    argparser.add_argument("--n_neg", type=int, default=20)

    argparser.add_argument("--samples_file", type=str)

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print '\n{}\n'.format(args)
    main()

