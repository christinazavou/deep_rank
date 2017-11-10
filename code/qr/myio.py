import gzip
from collections import Counter
from nn import EmbeddingLayer
import random
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.strip().split()
            body = body.strip().split()
            raw_corpus[id] = (title, body)
    print("{} empty titles ignored.\n".format(empty_cnt))

    # print ' raw_corpus keys :\n', raw_corpus.keys()[0:3]
    # print ' raw_corpus values :\n', raw_corpus.values()[0:3]
    return raw_corpus


def map_corpus(raw_corpus, embedding_layer, max_len=100):
    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len])
        # if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item

    # print ' ids corpus keys : \n', ids_corpus.keys()[0:3]
    # print ' ids corpus values : \n', ids_corpus.values()[0:3]
    return ids_corpus


def create_idf_weights(corpus_path, embedding_layer):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), binary=False)

    lst = []
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            lst.append(title)
            lst.append(body)
    vectorizer.fit_transform(lst)

    idfs = vectorizer.idf_  # global term weights - idf vector
    avg_idf = sum(idfs)/(len(idfs)+0.0)/4.0
    weights = np.array([avg_idf for i in xrange(embedding_layer.n_V)], dtype=np.float32)
    vocab_map = embedding_layer.vocab_map
    for word, idf_value in zip(vectorizer.get_feature_names(), idfs):
        id = vocab_map.get(word, -1)
        if id != -1:
            weights[id] = idf_value
    return tf.Variable(weights, name="word_weights", dtype=tf.float32)


def read_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = []  # i.e. ( ('421122', ['502523', '178532', '372151', '285015',...], [1, 1, 0, 1, ...]), (...), ... )
    with open(path) as fin:
        for line in fin:
            parts = line.split("\t")
            pid, pos, neg = parts[:3]
            pos = pos.split()
            neg = neg.split()
            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1):
                # if no positives skip the instance - it's not useful if no positives
                # if positives are more than prune_positive_counts value skip the instance - it's not useful if
                # too many positives
                continue
            if K_neg != -1:
                # shuffle negatives so that we don't feed in the network with same order always - avoid overfitting
                random.shuffle(neg)
                neg = neg[:K_neg]
            s = set()
            qids = []
            qlabels = []
            for q in neg:
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    s.add(q)
            for q in pos:
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    s.add(q)
            # print 'append: {}\n'.format((pid, qids, qlabels))
            lst.append((pid, qids, qlabels))
    return lst


def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    # returns actually a list of tuples (titles, bodies, qlabels) - each tuple defined for one eval instance
    # i.e. ([21x100], [21x100], [21x1]) if 10 pos and 20 neg
    # and so tuples can have different title/body shapes
    lst = []
    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        # print 't ', titles.shape, ' b ', bodies.shape
        lst.append((titles, bodies, np.array(qlabels, dtype="int32")))
    return lst


def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):

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
    batches = []

    query_per_triple = []

    for u in xrange(N):
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
        pid = pid2id[pid]
        pos = [pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id]
        neg = [pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id]

        query_per_triple.append(
            range(len(triples), len(triples)+len(pos))+[-1 for x in range(len(pos), 10)])  # because prune_pos_cnt = 10

        triples += [[pid, x]+neg for x in pos]

        # print 'add to triples: \n', [[pid, x]+neg for x in pos]

        # print cnt, len(pos), len(triples)

        if cnt == batch_size or u == N-1:
            # print cnt == batch_size, cnt, len(triples)
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
            # titles.shape = [max_title_len x batch_size], bodies.shape = [max_body_len x batch_size]

            triples = create_hinge_batch(triples)
            query_per_triple = np.array(query_per_triple, np.int32)

            batches.append((titles, bodies, triples, query_per_triple))

            titles = []
            bodies = []
            triples = []
            pid2id = {}
            cnt = 0

            query_per_triple = []

    return batches


def create_one_batch(titles, bodies, padding_id, pad_left):
    # each batch has its own questions with its own max-length ...
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    # pad according to those max lengths
    if pad_left:
        titles = np.column_stack(
            [np.pad(x, (max_title_len-len(x), 0), 'constant', constant_values=padding_id) for x in titles]
        )
        bodies = np.column_stack(
            [np.pad(x, (max_body_len-len(x), 0), 'constant', constant_values=padding_id) for x in bodies]
        )
    else:
        titles = np.column_stack(
            [np.pad(x, (0, max_title_len-len(x)), 'constant', constant_values=padding_id) for x in titles]
        )
        bodies = np.column_stack(
            [np.pad(x, (0, max_body_len-len(x)), 'constant', constant_values=padding_id) for x in bodies]
        )
    return titles, bodies


def create_hinge_batch(triples):
    # an instance in the triples list (i.e. a triple) is a list of: pid, qids (similar and not)
    # regularly one batch that can specify hinge loss has 22 question ids
    # so we create constant sized batches with 22 length x batch length
    max_len = max(len(x) for x in triples)
    triples = np.vstack(
        [np.pad(x, (0, max_len-len(x)), 'edge') for x in triples]
    ).astype('int32')
    return triples
