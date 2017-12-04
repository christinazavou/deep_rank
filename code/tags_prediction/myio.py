import gzip
import sys
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import tensorflow as tf
import pickle


def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()


def read_corpus(path, with_tags=False, test=-1):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        if with_tags:
            for test_id, line in enumerate(fin):
                id, title, body, tags = line.split("\t")
                if len(title) == 0:
                    print id
                    empty_cnt += 1
                    continue
                title = title.strip().split()
                body = body.strip().split()
                tags = tags.strip().split(u', ')
                raw_corpus[id] = (title, body, tags)
                if test_id == test:
                    break
        else:
            for test_id, line in enumerate(fin):
                id, title, body = line.split("\t")
                if len(title) == 0:
                    print id
                    empty_cnt += 1
                    continue
                title = title.strip().split()
                body = body.strip().split()
                raw_corpus[id] = (title, body)
                if test_id == test:
                    break
    print("{} empty titles ignored.\n".format(empty_cnt))

    # print ' raw_corpus keys :\n', raw_corpus.keys()[0:3]
    # print ' raw_corpus values :\n', raw_corpus.values()[0:3]
    return raw_corpus


def map_corpus(raw_corpus, embedding_layer, tags_selected, max_len=100):
    # keys in ids_corpus_tags are int type and in raw_corpus are str type

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=tags_selected)
    mlb.fit(['no_tag'])

    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():
        item = (
            embedding_layer.map_to_ids(pair[0], filter_oov=True),
            embedding_layer.map_to_ids(pair[1], filter_oov=True)[:max_len],
            mlb.transform([set(pair[2]) & set(tags_selected)])[0]
        )
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


def create_batches(df, ids_corpus, data_type, batch_size, padding_id, perm=None, N_neg=20, samples_file=None):

    samples_dict = None
    if samples_file:
        samples_dict = pickle.load(open(samples_file, 'rb'))

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
    tuples = []
    # tag_samples = []

    def transform(counter, x, length):
        return ((counter - 1) * length) + x

    for u in xrange(N):
        i = perm[u]
        q_id = data_ids[i]
        title, body, tag = ids_corpus[str(q_id)]
        cnt += 1
        titles.append(title)
        bodies.append(body)
        tag_labels.append(tag)

        q_positive_idx = [idx for idx, label in enumerate(tag) if label == 1]
        q_negative_idx = [idx for idx, label in enumerate(tag) if label == 0]
        q_positive_ids = [transform(cnt, idx, tag.shape[0]) for idx in q_positive_idx]
        q_negative_ids = [transform(cnt, idx, tag.shape[0]) for idx in q_negative_idx]
        if samples_dict:
            neg_samples, neg_sampled_tags = samples_dict[q_id]  # 100 tags
            neg_samples = list(neg_samples)
            q_negative_idx = neg_samples
            neg_samples = [transform(cnt, idx, tag.shape[0]) for idx in q_negative_idx]
            assert set(neg_samples) < set(q_negative_ids)
            q_negative_ids = neg_samples
        np.random.shuffle(q_negative_ids)
        q_negative_ids = q_negative_ids[:N_neg]  # consider only 20 negatives
        tuples += [[pid]+q_negative_ids for pid in q_positive_ids]  # if no positives, no tuples added
        # tag_samples.append(q_positive_idx + q_negative_idx)

        if cnt == batch_size or u == N-1:
            titles, bodies, tag_labels = create_one_batch(titles, bodies, tag_labels, padding_id)
            tuples = create_hinge_batch(tuples)
            # tag_samples = create_hinge_batch(tag_samples)
            yield titles, bodies, tag_labels, tuples

            titles, bodies, tag_labels = [], [], []
            cnt = 0
            tuples = []
            # tag_samples = []


def create_hinge_batch(triples):
    # an instance in the triples list (i.e. a triple) is a list of: pid, qids (similar and not)
    # regularly one batch that can specify hinge loss has 22 question ids
    # so we create constant sized batches with 22 length x batch length
    max_len = max(len(x) for x in triples)
    triples = np.vstack(
        [np.pad(x, (0, max_len-len(x)), 'edge') for x in triples]
    ).astype('int32')
    return triples


def create_one_batch(titles, bodies, tag_labels, padding_id):
    # each batch has its own questions with its own max-length ...
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    # pad according to those max lengths
    titles = np.column_stack(
        [np.pad(x, (0, max_title_len-len(x)), 'constant', constant_values=padding_id) for x in titles]
    )
    bodies = np.column_stack(
        [np.pad(x, (0, max_body_len-len(x)), 'constant', constant_values=padding_id) for x in bodies]
    )
    tag_labels = np.stack(tag_labels)
    return titles, bodies, tag_labels


def create_cross_val_batches(df, ids_corpus, batch_size, padding_id, perm=None, N_neg=20, samples_file=None):

    samples_dict = None
    if samples_file:
        samples_dict = pickle.load(open(samples_file, 'rb'))

    data_ids = df['id'].values

    if perm is None:  # if no given order (i.e. perm), make a shuffle-random one.
        perm = range(len(data_ids))
        random.shuffle(perm)

    N = len(data_ids)

    # for one batch:
    cnt = 0
    titles, bodies, tag_labels = [], [], []
    tuples = []
    batches = []

    def transform(counter, x, length):
        return ((counter - 1) * length) + x

    for u in xrange(N):
        i = perm[u]
        q_id = data_ids[i]
        title, body, tag = ids_corpus[str(q_id)]
        cnt += 1
        titles.append(title)
        bodies.append(body)
        tag_labels.append(tag)
        q_positive_idx = [idx for idx, label in enumerate(tag) if label == 1]
        q_negative_idx = [idx for idx, label in enumerate(tag) if label == 0]
        q_positive_ids = [transform(cnt, idx, tag.shape[0]) for idx in q_positive_idx]
        q_negative_ids = [transform(cnt, idx, tag.shape[0]) for idx in q_negative_idx]
        if samples_dict:
            neg_samples, neg_sampled_tags = samples_dict[q_id]  # 100 tags
            neg_samples = list(neg_samples)
            q_negative_idx = neg_samples
            neg_samples = [transform(cnt, idx, tag.shape[0]) for idx in q_negative_idx]
            assert set(neg_samples) < set(q_negative_ids)
            q_negative_ids = neg_samples
        np.random.shuffle(q_negative_ids)
        q_negative_ids = q_negative_ids[:N_neg]  # consider only 20 negatives
        tuples += [[pid]+q_negative_ids for pid in q_positive_ids]  # if no positives, no tuples added

        if cnt == batch_size or u == N-1:
            titles, bodies, tag_labels = create_one_batch(titles, bodies, tag_labels, padding_id)
            tuples = create_hinge_batch(tuples)
            batches.append((titles, bodies, tag_labels, tuples))

            titles, bodies, tag_labels = [], [], []
            cnt = 0
            tuples = []

    total_batches = len(batches)
    train_total = total_batches * 0.8
    dev_total = train_total * 0.2
    dev_batches = batches[0: int(dev_total)]
    train_batches = batches[int(dev_total): int(train_total)]
    test_batches = batches[int(train_total):]
    return train_batches, dev_batches, test_batches
