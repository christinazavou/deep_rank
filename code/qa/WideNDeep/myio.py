import random
import numpy as np


def read_eval_annotations(path):
    lst = []
    with open(path) as fin:
        for line in fin:
            pid, pos, cand, scores, vectors = line.split("\t")

            pos = pos.split()
            cand = cand.split()
            vectors = vectors.split(', ')
            vectors = [np.array([float(v) for v in vec.split()]) for vec in vectors]

            if len(pos) == 0:
                continue

            s = set()
            qids = []
            qlabels = []
            qvectors = []
            for q, qvec in zip(cand, vectors):
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    qvectors.append(qvec)
                    s.add(q)
            lst.append((pid, qids, qlabels, qvectors))
    return lst


def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = []
    for pid, qids, qlabels, qvectors in data:
        titles = []
        bodies = []
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), np.array(qvectors, dtype="float32")))
    return lst


def create_one_batch(titles, bodies, padding_id, pad_left):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
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


def read_train_annotations(path, K_neg=20, prune_pos_cnt=10):
    lst = []  # i.e. ( ('421122', ['502523', '178532', '372151', '285015',...], [1, 1, 0, 1, ...]), (...), ... )
    with open(path) as fin:
        for line in fin:

            pid, pos, neg, pos_vectors, neg_vectors = line.split("\t")

            pos = pos.split()
            neg = neg.split()
            pos_vectors = pos_vectors.split(', ')
            pos_vectors = [np.array([float(v) for v in vec.split()]) for vec in pos_vectors]

            neg_vectors = neg_vectors.split(', ')
            neg_vectors = [np.array([float(v) for v in vec.split()]) for vec in neg_vectors]

            if len(pos) == 0 or (len(pos) > prune_pos_cnt and prune_pos_cnt != -1):
                continue

            perm = range(len(neg[:K_neg]))
            random.shuffle(perm)
            neg = [neg[i] for i in perm[:K_neg]]
            neg_vectors = [neg_vectors[i] for i in perm[:K_neg]]

            s = set()
            qids = []
            qlabels = []
            qvectors = []
            for q, qvec in zip(neg, neg_vectors):
                if q not in s:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    qvectors.append(qvec)
                    s.add(q)
            for q, qvec in zip(pos, pos_vectors):
                if q not in s:
                    qids.append(q)
                    qlabels.append(1)
                    qvectors.append(qvec)
                    s.add(q)
            lst.append((pid, qids, qlabels, qvectors))
    return lst


def create_train_batches(ids_corpus, data, batch_size, padding_id, perm=None, pad_left=True):

    if perm is None:
        perm = range(len(data))
        random.shuffle(perm)

    N = len(data)

    cnt = 0
    pid2id = {}
    titles = []
    bodies = []
    triples = []
    batches = []

    vectors_dict = {}

    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels, qvectors = data[i]

        if pid not in ids_corpus:
            continue

        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if id not in ids_corpus:
                    continue
                pid2id[id] = len(titles)
                t, b = ids_corpus[id]
                titles.append(t)
                bodies.append(b)

        for id, vec in zip(qids, qvectors):
            vectors_dict[(pid2id[pid], pid2id[id])] = vec

        pid = pid2id[pid]
        pos = [pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id]
        neg = [pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id]
        triples += [[pid, x]+neg for x in pos]

        if cnt == batch_size or u == N-1:
            titles, bodies = create_one_batch(titles, bodies, padding_id, pad_left)

            triples = create_hinge_batch(triples)

            vectors = np.array(
                [[vectors_dict[(triple[0], x)] for x in triple[1:]] for triple in triples]
            )

            batches.append((titles, bodies, triples, vectors))

            titles = []
            bodies = []
            triples = []
            pid2id = {}
            cnt = 0

    return batches


def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack(
        [np.pad(x, (0, max_len-len(x)), 'edge') for x in triples]
    ).astype('int32')
    return triples

