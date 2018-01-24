import argparse
import myio
import sys
from utils import load_embedding_iterator, create_embedding_layer
import time
import os
from datetime import datetime


def main():
    print 'Starting at: {}\n'.format(datetime.now())
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = create_embedding_layer(
        n_d=200,
        embs=load_embedding_iterator(args.embeddings),
        only_words=False if args.use_embeddings else True,
        trainable=args.trainable
    )
    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.reweight:
        weights = myio.create_idf_weights(args.corpus, embedding_layer)

    if args.layer.lower() == "lstm":
        from models import LstmQR as Model
    elif args.layer.lower() in ["bilstm", "bigru"]:
        from models import BiRNNQR as Model
    elif args.layer.lower() == "cnn":
        from models import CnnQR as Model
    elif args.layer.lower() == "gru":
        from models import GruQR as Model
    else:
        raise Exception("no correct layer given")

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus, dev, padding_id, pad_left=False)
    if args.test:
        test = myio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus, test, padding_id, pad_left=False)

    model = Model(args, embedding_layer, weights=weights if args.reweight else None)
    model.ready()

    print 'total (non) trainable params: ', model.num_parameters()

    if args.load_pre_trained_part:
        # need to remove the old assigns to embeddings
        model.init_assign_ops = model.load_pre_trained_part(args.load_pre_trained_part)
    print '\nmodel init_assign_ops: {}\n'.format(model.init_assign_ops)

    if args.train:
        start_time = time.time()
        train = myio.read_annotations(args.train)
        train_batches = myio.create_batches(
            ids_corpus, train, args.batch_size, padding_id, pad_left=False
        )

        print("{} to create batches\n".format(time.time()-start_time))
        print("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))

        model.train_model(
            ids_corpus,
            train,
            dev=dev if args.dev else None,
            test=test if args.test else None
        )
    print '\nEnded at: {}'.format(datetime.now())


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--load_pre_trained_part", type=str, default="")

    argparser.add_argument("--use_embeddings", type=int, default=1)  # refers to word embeddings
    argparser.add_argument("--trainable", type=int, default=1)  # refers to word embeddings
    argparser.add_argument("--load_only_embeddings", type=int, default=0)  # refers to word embeddings

    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)
    # average 1 = mean pooling for any layer
    # average 0 = last pooling for RNNs max pooling for CNN
    # average 2 = max pooling for RNNs
    argparser.add_argument("--average", type=int, default=0)
    argparser.add_argument("--batch_size", type=int, default=40)
    argparser.add_argument("--performance", type=str, default='MAP')  # 'MRR' or 'MAP'
    # argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--patience", type=int, default=8)
    argparser.add_argument("--normalize", type=int, default=1)
    argparser.add_argument("--reweight", type=int, default=0)
    argparser.add_argument("--layer", type=str, default="lstm")
    argparser.add_argument("--concat", type=int, default=0)
    argparser.add_argument("--loss", type=int, default=0)
    argparser.add_argument("--weight", type=float, default=1.)
    argparser.add_argument("--mlp_dim", type=int, default=0)
    argparser.add_argument("--mlp_dim2", type=int, default=0)

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print '\n{}\n'.format(args)
    main()
