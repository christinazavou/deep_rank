import numpy as np
import random


def create_eval_batches(ids_corpus, data, padding_id, N_neq=20):
    lst = []

    def transform(counter, x, length):
        return ((counter - 1) * length) + x

    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        tag_labels = []
        cnt_q = 0
        tuples = []
        for id in [pid]+qids:
            cnt_q += 1
            title, body, tag = ids_corpus[str(id)]
            titles.append(title)
            bodies.append(body)
            tag_labels.append(tag)

            q_positive_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 1]
            q_negative_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 0]
            np.random.shuffle(q_negative_ids)
            q_negative_ids = q_negative_ids[:N_neq]  # consider only 20 negatives
            tuples += [[p_id] + q_negative_ids for p_id in q_positive_ids]

        tuples = create_hinge_batch(tuples)
        titles, bodies, tag_labels = create_one_batch(titles, bodies, tag_labels, padding_id)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), tag_labels, tuples))
    return lst


def create_batches(ids_corpus, data, batch_size, padding_id, perm=None, N_neq=20):

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
    tag_labels = []

    tuples = []
    cnt_q = 0
    query_per_triple = []

    def transform(counter, x, length):
        return ((counter - 1) * length) + x

    for u in xrange(N):
        i = perm[u]
        pid, qids, qlabels = data[i]
        if str(pid) not in ids_corpus:  # use only known pid's from corpus seen already
            continue
        cnt += 1
        for id in [pid] + qids:
            if id not in pid2id:
                if str(id) not in ids_corpus:  # use only known qids from corpus seen already
                    continue
                pid2id[id] = len(titles)
                title, body, tag = ids_corpus[id]
                titles.append(title)
                bodies.append(body)
                tag_labels.append(tag)
                cnt_q += 1

                q_positive_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 1]
                q_negative_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 0]
                np.random.shuffle(q_negative_ids)
                q_negative_ids = q_negative_ids[:N_neq]  # consider only 20 negatives
                tuples += [[p_id] + q_negative_ids for p_id in q_positive_ids]

        pid = pid2id[pid]
        pos = [pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id]
        neg = [pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id]
        query_per_triple.append(
            range(len(triples), len(triples)+len(pos))+[-1 for x in range(len(pos), 10)])  # because prune_pos_cnt = 10

        triples += [[pid, x]+neg for x in pos]

        if cnt == batch_size or u == N-1:
            titles, bodies, tag_labels = create_one_batch(titles, bodies, tag_labels, padding_id)
            triples = create_hinge_batch(triples)
            query_per_triple = np.array(query_per_triple, np.int32)
            tuples = create_hinge_batch(tuples)
            batches.append((titles, bodies, triples, query_per_triple, tag_labels, tuples))

            titles = []
            bodies = []
            triples = []
            tag_labels = []
            pid2id = {}
            cnt = 0

            tuples = []
            cnt_q = 0
            query_per_triple = []

    return batches


def create_one_batch(titles, bodies, tag_labels, padding_id):
    max_title_len = max(1, max(len(x) for x in titles))
    max_body_len = max(1, max(len(x) for x in bodies))
    # pad right
    titles = np.column_stack(
        [np.pad(x, (0, max_title_len-len(x)), 'constant', constant_values=padding_id) for x in titles]
    )
    bodies = np.column_stack(
        [np.pad(x, (0, max_body_len-len(x)), 'constant', constant_values=padding_id) for x in bodies]
    )
    tag_labels = np.stack(tag_labels)
    return titles, bodies, tag_labels


def create_hinge_batch(triples):
    max_len = max(len(x) for x in triples)
    triples = np.vstack(
        [np.pad(x, (0, max_len-len(x)), 'edge') for x in triples]
    ).astype('int32')
    return triples
