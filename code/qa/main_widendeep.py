import numpy as np
import scipy.sparse
from nltk.util import ngrams
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import random
from sklearn.metrics.pairwise import cosine_similarity
import myio
import os
import time
import sys
from utils import load_embedding_iterator


from main_model_widendeep import Model


def make_features_dicts(ids_corpus, raw_corpus):
    if os.path.isfile('/home/christina/Documents/Thesis/deep_rank/n_gram_vectors.npy.npz') and\
            os.path.isfile('/home/christina/Documents/Thesis/deep_rank/pos_tag_n_gram_vectors.npy.npz'):

        n_gram_vec = scipy.sparse.load_npz('/home/christina/Documents/Thesis/deep_rank/n_gram_vectors.npy.npz')
        n_gram_pos_tag_vec = scipy.sparse.load_npz('/home/christina/Documents/Thesis/deep_rank/pos_tag_n_gram_vectors.npy.npz')

    else:
        s_time = time.time()
        ngrams_as_sent = []
        for i, (id, (title, body)) in enumerate(ids_corpus.iteritems()):

            # if i == 10:  # TEST
            #     break

            n_grams = [str(w) for w in title] + [str(w) for w in body]
            for n in [2, 3, 4]:
                n_grams += ["_".join([str(w) for w in gr]) for gr in ngrams(title, n)]
                n_grams += ["_".join([str(w) for w in gr]) for gr in ngrams(title, n)]

            ngrams_as_sent.append(" ".join(n_grams))

        vectorizer = TfidfVectorizer(encoding='utf8', max_df=1.0, min_df=5, max_features=None, norm='l2', use_idf=False)
        n_gram_vec = vectorizer.fit_transform(ngrams_as_sent)
        scipy.sparse.save_npz('/home/christina/Documents/Thesis/deep_rank/n_gram_vectors.npy', n_gram_vec)

        ngrams_as_sent = []
        for i, (id, (title, body)) in enumerate(raw_corpus.iteritems()):

            # if i == 10:  # TEST
            #     break

            title_pos_tags = [w[1] for w in nltk.pos_tag(title)]
            body_pos_tags = [w[1] for w in nltk.pos_tag(body)]
            pos_tags_n_grams = title_pos_tags + body_pos_tags
            for n in [2, 3, 4]:
                pos_tags_n_grams += ["_".join([w for w in gr]) for gr in ngrams(title_pos_tags, n)]
                pos_tags_n_grams += ["_".join([w for w in gr]) for gr in ngrams(body_pos_tags, n)]

            ngrams_as_sent.append(" ".join(pos_tags_n_grams))

        vectorizer = TfidfVectorizer(encoding='utf8', max_df=1.0, min_df=5, max_features=None, norm='l2', use_idf=False)
        n_gram_pos_tag_vec = vectorizer.fit_transform(ngrams_as_sent)
        scipy.sparse.save_npz('/home/christina/Documents/Thesis/deep_rank/pos_tag_n_gram_vectors.npy', n_gram_pos_tag_vec)
        print 'took ', time.time() - s_time  # 1286 sec i.e. 20 min

    return n_gram_vec, n_gram_pos_tag_vec


def create_eval_batches(ids_corpus, data, padding_id, f1_vectors, f2_vectors, pad_left):
    # returns actually a list of tuples (titles, bodies, qlabels) - each tuple defined for one eval instance
    # i.e. ([21x100], [21x100], [21x1]) if 10 pos and 20 neg
    # and so tuples can have different title/body shapes
    lst = []

    ids = ids_corpus.keys()

    for i, (pid, qids, qlabels) in enumerate(data):

        if i % 100 == 0:
            print 'i is ', i
        if i == 10:  # TEST
            break

        titles = []
        bodies = []

        vec_ids = []

        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)

            vec_ids.append(ids.index(id))

        titles, bodies = myio.create_one_batch(titles, bodies, padding_id, pad_left)
        # print 't ', titles.shape, ' b ', bodies.shape, ' vec_ids ', len(vec_ids)

        features = create_feature_array(f1_vectors[vec_ids], f2_vectors[vec_ids])

        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), features))
    return lst


def create_feature_array(f1_mat, f2_mat):
    features = np.zeros((f1_mat.shape[0]-1, 2))
    features[:, 0] = cosine_similarity(f1_mat[0], f1_mat[1:])[0]
    features[:, 1] = cosine_similarity(f2_mat[0], f2_mat[1:])[0]
    # todo: jaccard of non vectors
    return features


def create_batches(ids_corpus, data, batch_size, padding_id, f1_vectors, f2_vectors, perm=None, pad_left=True):

    # returns a list of batches where each batch is a list of (titles, bodies and some hidge_loss_batches ids to
    # indicate relations/cases happening in current titles and bodies...)

    if perm is None:  # if no given order (i.e. perm), make a shuffle-random one.
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)

    # for one batch:
    cnt = 0
    pid2id = {}
    titles = []
    bodies = []
    triples = []

    vec_ids = []
    ids = ids_corpus.keys()

    batches = []

    for u in xrange(N):
        if u % 100 == 0:
            print 'u is ', u
        if u == 100:  # TAKES TOO LONG..RUN ON  15000
            break
        i = perm[u]
        pid, qids, qlabels = data[i]
        if pid not in ids_corpus:  # use only known pid's from corpus seen already
            continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus:  # use only known qids from corpus seen already
                    continue
                pid2id[id] = len(titles)
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)

                vec_ids.append(ids.index(id))

        pid = pid2id[pid]
        pos = [pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id]
        neg = [pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id]
        triples += [[pid, x]+neg for x in pos]
        # print 'add to triples: \n', [[pid, x]+neg for x in pos]

        if cnt == batch_size or u == N-1:

            titles, bodies = myio.create_one_batch(titles, bodies, padding_id, pad_left)
            # titles.shape = [max_title_len x batch_size], bodies.shape = [max_body_len x batch_size]

            triples = myio.create_hinge_batch(triples)

            features = create_feature_mat(f1_vectors[vec_ids], f2_vectors[vec_ids], triples)

            batches.append((titles, bodies, triples, features))

            titles = []
            bodies = []
            triples = []
            pid2id = {}
            cnt = 0

            vec_ids = []

    return batches


def create_feature_mat(f1_mat, f2_mat, triples):
    features = np.zeros((triples.shape[0], triples.shape[1]-1, 2))
    for i, triple in enumerate(triples):
        features[i, :, 0] = cosine_similarity(f1_mat[triple[0]], f1_mat[triple[1:]])
        features[i, :, 1] = cosine_similarity(f2_mat[triple[0]], f2_mat[triple[1:]])
    # todo: jaccard of non vectors
    return features


def main(args):
    raw_corpus = myio.read_corpus(args.corpus)
    embedding_layer = myio.create_embedding_layer(
                raw_corpus,
                n_d=args.hidden_dim,
                cut_off=args.cut_off,
                embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
            )
    ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    n_gram_vectors, pos_tag_n_gram_vectors = make_features_dicts(ids_corpus, raw_corpus)

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = create_eval_batches(ids_corpus, dev, padding_id, n_gram_vectors, pos_tag_n_gram_vectors, pad_left=not args.average)
    if args.test:
        test = myio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus, test, padding_id, n_gram_vectors, pos_tag_n_gram_vectors, pad_left=not args.average)

    model = Model(args, embedding_layer)
    model.ready()

    assign_ops = model.load_trained_vars(args.load_pretrain) if args.load_pretrain else None

    if args.train:
        start_time = time.time()
        train = myio.read_annotations(args.train)
        train_batches = create_batches(
            ids_corpus, train, args.batch_size, padding_id, n_gram_vectors, pos_tag_n_gram_vectors, pad_left=not args.average
        )
        print("{} to create batches\n".format(time.time()-start_time))
        print("{} batches, {} tokens in total, {} triples in total\n".format(
                len(train_batches),
                sum(len(x[0].ravel())+len(x[1].ravel()) for x in train_batches),
                sum(len(x[2].ravel()) for x in train_batches)
            ))

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
    # argparser.add_argument("--learning", type=str, default="adam")
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--l2_reg", type=float, default=1e-5)
    argparser.add_argument("--activation", "-act", type=str, default="tanh")
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--dropout", type=float, default=0.0)
    argparser.add_argument("--max_epoch", type=int, default=50)
    argparser.add_argument("--normalize", type=int, default=1)
    # argparser.add_argument("--reweight", type=int, default=1)

    argparser.add_argument("--load_pretrain", type=str, default="")

    timestamp = str(int(time.time()))
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "runs", timestamp)

    argparser.add_argument("--save_dir", type=str, default=out_dir)

    args = argparser.parse_args()
    print args
    print ""
    main(args)