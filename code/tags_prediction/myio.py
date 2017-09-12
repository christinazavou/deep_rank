import gzip
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf


def read_corpus(path, with_tags=False):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        if with_tags:
            for line in fin:
                id, title, body, tags = line.split("\t")
                if len(title) == 0:
                    print id
                    empty_cnt += 1
                    continue
                title = title.strip().split()
                body = body.strip().split()
                tags = tags.strip().split(u', ')
                raw_corpus[id] = (title, body, tags)
        else:
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


def map_corpus(raw_corpus, embedding_layer, ids_corpus_tags, max_len=100):
    # keys in ids_corpus_tags are int type and in raw_corpus are str type
    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len],
                ids_corpus_tags[int(id)])
        # if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item

    # print ' ids corpus keys : \n', ids_corpus.keys()[0:3]
    # print ' ids corpus values : \n', ids_corpus.values()[0:3]
    return ids_corpus


def map_corpus2(raw_corpus, embedding_layer, tags_selected, max_len=100):
    # keys in ids_corpus_tags are int type and in raw_corpus are str type

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=tags_selected)
    mlb.fit(['no_tag'])

    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():
        item = (embedding_layer.map_to_ids(pair[0], filter_oov=True),
                embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len],
                mlb.transform([set(pair[2]) & set(tags_selected)]))
        # if len(item[0]) == 0:
        #    say("empty title after mapping to IDs. Doc No.{}\n".format(id))
        #    continue
        ids_corpus[id] = item

    # print ' ids corpus keys : \n', ids_corpus.keys()[0:3]
    # print ' ids corpus values : \n', ids_corpus.values()[0:3]
    return ids_corpus


def create_idf_weights(corpus_path, embedding_layer, with_tags=False):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), binary=False)

    lst = []
    fopen = gzip.open if corpus_path.endswith(".gz") else open
    with fopen(corpus_path) as fin:
        if with_tags:
            for line in fin:
                id, title, body, tags = line.split("\t")
                lst.append(title)
                lst.append(body)
        else:
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


def make_tag_labels(df, tags_selected):
    df = df[['id', 'title', 'body']+tags_selected]
    ids_corpus_tags = {}
    for idx, row in df.iterrows():
        ids_corpus_tags[row['id']] = row[tags_selected].values
    return ids_corpus_tags


def create_batches(df, ids_corpus, data_type, batch_size, padding_id, perm=None, pad_left=True):

    # returns a list of batches where each batch is a list of (titles, bodies tags-as-np-array)

    # df ids are int ids_corpus ids are str

    df = df[df['type'] == data_type]

    data_ids = df['id'].values

    if perm is None:  # if no given order (i.e. perm), make a shuffle-random one.
        perm = range(len(data_ids))
        random.shuffle(perm)

    N = len(data_ids)

    # for one batch:
    cnt = 0
    titles, bodies, tag_labels = [], [], []
    batches = []

    for u in xrange(N):
        i = perm[u]
        q_id = data_ids[i]
        title, body, tag = ids_corpus[str(q_id)]
        cnt += 1
        titles.append(title)
        bodies.append(body)
        tag_labels.append(tag)

        if cnt == batch_size or u == N-1:
            titles, bodies, tag_labels = create_one_batch(titles, bodies, tag_labels, padding_id, pad_left)
            batches.append((titles, bodies, tag_labels))

            titles, bodies, tag_labels = [], [], []
            cnt = 0

    return batches


def create_one_batch(titles, bodies, tag_labels, padding_id, pad_left):
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
    tag_labels = np.stack(tag_labels)
    return titles, bodies, tag_labels

