import argparse
import os
import pickle
import sys
import time
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

from evaluation import Evaluation
import scipy.sparse
from sklearn.utils import shuffle

import logging
from Queue import Full
from multiprocessing import Queue, Pool
import itertools
import random
from multiprocessing import freeze_support

logging.basicConfig(level=logging.INFO)


def read_df(df_file, chunk_size=None, read_columns=None):
    if '.csv' in df_file:
        if chunk_size:
            return pd.read_csv(df_file, encoding='utf8', index_col=0, chunksize=chunk_size)
        else:
            if read_columns:
                return pd.read_csv(df_file, encoding='utf8', index_col=0, usecols=read_columns)
            else:
                return pd.read_csv(df_file, encoding='utf8', index_col=0)
    elif '.p' in df_file:
        if read_columns:
            return pd.read_pickle(df_file)[read_columns]
        else:
            return pd.read_pickle(df_file)
    else:
        raise Exception(' unknown pandas file {}'.format(df_file))


def get_data(df, labels, type_name=None):
    if type_name is not None:
        df = df[df['type'] == type_name]
    if args.truncate:
        return (df['title'] + u' ' + df['body_truncated']).values, df[labels].values
    else:
        return (df['title'] + u' ' + df['body']).values, df[labels].values


def pre_process(x):
    new_x = []
    st = PorterStemmer()
    for xi in x:
        new_x.append([st.stem(w) for w in word_tokenize(xi) if w not in stopwords.words('english')])
    return new_x


class MLRComponent(object):

    def __init__(self, model_file=None):
        self.model_file = model_file
        if model_file and os.path.isfile(self.model_file):
            self.clf = pickle.load(open(self.model_file, 'rb'))
            print 'mlr component loaded'
        else:
            self.clf = None

    def fit(self, x_train, y_train):
        if self.clf is None:
            start_time = time.time()
            print 'please wait while mlr component is trained'
            self.clf = OneVsRestClassifier(MultinomialNB(), n_jobs=1)
            self.clf.fit(x_train, y_train)
            pickle.dump(self.clf, open(self.model_file, 'wb'), protocol=2)
            print 'mlr component finished training after {} minutes'.format((time.time()-start_time) // 60)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)


class SimComponent(object):

    def __init__(self):
        self.tf_idf_trans = TfidfTransformer()
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        x_train = self.tf_idf_trans.fit_transform(x_train)
        self.x_train = x_train
        self.y_train = y_train

    def predict_proba(self, x_eval):
        x_eval = self.tf_idf_trans.transform(x_eval)
        y_scores = []  # per sample

        batch_size = x_eval.shape[0]/30
        top_50_per_sample = np.zeros((x_eval.shape[0], 50))
        for i in range(20):
            cos = cosine_similarity(self.x_train, x_eval[i*batch_size: (i+1)*batch_size]).T
            cos = np.argsort(cos, 1)[:, -50:]
            top_50_per_sample[(i*batch_size): (i+1)*batch_size, :] = cos
        top_50_per_sample = top_50_per_sample.astype(np.int32)

        # similarities = cosine_similarity(self.x_train, x_eval).T  # num_of_eval_samples x num_of_train_samples
        # assert similarities.shape[0] == x_eval.shape[0] and similarities.shape[1] == self.x_train.shape[0]

        # argsort gives the less similar first
        # top_50_per_sample = np.argsort(similarities, 1)[:, -50:]  # num_of_eval_samples x 50
        # assert top_50_per_sample.shape == x_eval.shape[0] and top_50_per_sample.shape[1] == 50

        for top_50 in top_50_per_sample:

            top_50_y = self.y_train[top_50, :]  # 50 x num_of_tags

            vote_per_tag = np.sum(top_50_y, 0)  # num_of_tags
            sum_per_not_tag = np.sum(vote_per_tag) - vote_per_tag  # num_of_tags
            score_per_tag = vote_per_tag.astype(np.float32) / (sum_per_not_tag.astype(np.float32) + 1e-9)  # num_of_tags

            y_scores.append(score_per_tag)

        return np.array(y_scores)  # num_of_samples x num_of_tags


class TagTermComponent(object):

    def __init__(self, model_file):
        self.model_file = model_file
        if os.path.isfile(model_file):
            self.affinity_term_tag = pickle.load(open(self.model_file, 'rb'))
            print 'tt component loaded'
        else:
            self.affinity_term_tag = None

    def fit(self, x_train, y_train):
        if self.affinity_term_tag is None:
            print 'please wait until affinity scores are calculated'
            start_time = time.time()
            # split into multiple matmul since memory consumption is huge
            idx_s = 0
            batch_size = 200
            n_term_tag = np.zeros((x_train.shape[1], y_train.shape[1]))
            while idx_s < x_train.shape[1]-1:
                idx_e = min(idx_s + batch_size, x_train.shape[1]-1)
                n_term_tag[idx_s: idx_e] = np.matmul(x_train[:, idx_s:idx_e].toarray().T, y_train)
                idx_s = idx_e
            n_tag = np.sum(y_train, 0)  # num_of_tags
            self.affinity_term_tag = n_term_tag / (n_tag + 1e-9)  # num_of_terms x num_of_tags
            pickle.dump(self.affinity_term_tag, open(self.model_file, 'wb'), protocol=2)
            print 'affinity scores calculated after {} minutes'.format((time.time()-start_time) // 60)

    def predict_proba(self, x):
        y_scores = []
        for xi in x:
            x_terms, _ = np.nonzero(xi)
            y_scores.append(1. - np.prod(1. - self.affinity_term_tag[x_terms, :], axis=0))
        return np.array(y_scores)


def evaluate(test_y, y_scores, verbose=0):
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    eval_labels = []
    for label in range(test_y.shape[1]):
        if (test_y[:, label] == np.ones(test_y.shape[0])).any():
            eval_labels.append(label)
    eval_samples = []
    for sample in range(test_y.shape[0]):
        if (test_y[sample, :] == np.ones(test_y.shape[1])).any():
            eval_samples.append(sample)

    test_y, y_scores = test_y[eval_samples, :], y_scores[eval_samples, :]
    test_y, y_scores = test_y[:, eval_labels], y_scores[:, eval_labels]

    ev = Evaluation(y_scores, None, test_y)

    if verbose:
        print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\tR@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\tUBP@5: {}\tUBP@10: {}\tMAP: {}\n'.format(
            ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10),
            ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10), ev.upper_bound(5), ev.upper_bound(10),
            ev.MeanAveragePrecision()
        )
    return ev.Recall(10)


def worker_eval(input_queue, result_queue):
    while True:
        logging.info("getting a new job")
        combo_rat10 = {}
        chunk_no, chunk, probs_mlr, probs_sim, probs_tt, targets = input_queue.get()
        for combo in chunk:
            [alpha, beta, gamma] = combo
            probs = alpha*probs_mlr + beta*probs_sim + gamma*probs_tt
            rat10 = evaluate(targets, probs)
            combo_rat10[(alpha, beta, gamma)] = rat10
        logging.info("processed chunk, queuing the result")
        result_queue.put(combo_rat10)


def chunk_serial(iterable, chunk_size):
    it = iter(iterable)
    while True:
        wrapped_chunk = [list(itertools.islice(it, int(chunk_size)))]
        if not wrapped_chunk[0]:
            break
        yield wrapped_chunk.pop()


def find_texts_parallel(probs_mlr, probs_sim, probs_tt, targets, workers=2):
    job_queue = Queue(2*workers)
    result_queue = Queue()

    pool = Pool(workers, worker_eval, (job_queue, result_queue,))
    queue_size = [0]  # integer can't be accessed in inner definition so list used
    all_combo_rat10 = {}

    def process_result_queue():
        """Clear the result queue, merging all intermediate results"""
        while not result_queue.empty():
            combos_rat10 = result_queue.get()

            for combo, rat10 in combos_rat10.iteritems():
                # add to the existing one. If we had one dict to update, each time it would keep the current text found
                all_combo_rat10.setdefault(combo, 0)
                all_combo_rat10[combo] += rat10

            queue_size[0] -= 1

    values = [
        [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
        [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    ]
    combos = list(itertools.product(*values))
    total_combos = len(combos)
    for chunk_no, chunk in enumerate(chunk_serial(combos, 400)):
        # put the chunk into the workers' input job queue
        chunk_put = False
        while not chunk_put:
            try:
                job_queue.put((chunk_no, chunk, probs_mlr, probs_sim, probs_tt, targets), block=False)
                chunk_put = True
                queue_size[0] += 1
                logging.info('PROGRESS: dispatched chunk {} = '
                             'combos up to {}, over {} combos, outstanding queue size {}'.
                             format(chunk_no, chunk_no * 400 + len(chunk), total_combos, queue_size[0]))
            except Full:
                # in case the input job queue is full, keep clearing the result queue to make sure we don't deadlock
                process_result_queue()

        process_result_queue()

    while queue_size[0] > 0:  # wait for all outstanding jobs to finish
        process_result_queue()

    pool.terminate()

    return all_combo_rat10


def effective_weights(x_train, y_train, x_dev, y_dev, njobs=3):   # , sample_size, performance

    mlr = MLRComponent(args.model_file_mlr)
    mlr.fit(x_train, y_train)
    mlr_probs = mlr.predict_proba(x_dev)
    # evaluate(y_dev, mlr_probs)

    sim = SimComponent()
    sim.fit(x_train, y_train)
    sim_probs = sim.predict_proba(x_dev)
    # evaluate(y_dev, sim_probs)

    tt = TagTermComponent(args.model_file_tt)
    tt.fit(x_train, y_train)
    tt_probs = tt.predict_proba(x_dev)
    # evaluate(y_dev, tt_probs)

    if args.cross_val:
        start_time = time.time()
        combo_results = find_texts_parallel(mlr_probs, sim_probs, tt_probs, y_dev, workers=njobs)

        print 'took {} minutes for all combinations '.format((time.time()-start_time) // 60)
        sorted_results = sorted(combo_results.items(), key=lambda x: x[1])[::-1]
        print '\n', sorted_results, '\n'
        print 'best combo: {}'.format(sorted_results[0])
        return sorted_results[0][0], tt, mlr, sim

    return (0.1, 1.0, 0), tt, mlr, sim


def main():

    df = read_df(args.df_path)
    df = df.fillna(u'')
    label_tags = pickle.load(open(args.tags_file, 'rb'))
    print '\nloaded {} tags'.format(len(label_tags))

    if not args.cross_val:
        x_train, y_train = get_data(df, label_tags, 'train')
        x_dev, y_dev = get_data(df, label_tags, 'dev')
        x_test, y_test = get_data(df, label_tags, 'test')
    else:
        x, y = get_data(df, label_tags)
        x, y = shuffle(x, y)
        total = len(x)
        train_total = total * 0.8
        dev_total = train_total * 0.2
        x_dev, y_dev = x[0: int(dev_total)], y[0: int(dev_total)]
        x_train, y_train = x[int(dev_total): int(train_total)], y[int(dev_total): int(train_total)]
        x_test, y_test = x[int(train_total):], y[int(train_total):]
    del df

    bow_vec = CountVectorizer(min_df=20, max_df=1.0, ngram_range=(1, 1), binary=True)
    x_train = bow_vec.fit_transform(x_train)
    x_dev = bow_vec.transform(x_dev)
    x_test = bow_vec.transform(x_test)
    print 'shapes ', x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape, y_test.shape

    (alpha, beta, gama), tt, mlr, sim = effective_weights(x_train, y_train, x_dev, y_dev, args.njobs)

    mlr_probs = mlr.predict_proba(x_dev)
    sim_probs = sim.predict_proba(x_dev)
    tt_probs = tt.predict_proba(x_dev)
    probs = alpha*mlr_probs + beta*sim_probs + gama*tt_probs
    print 'dev evaluation: '
    evaluate(y_dev, probs, 1)

    mlr_probs = mlr.predict_proba(x_test)
    sim_probs = sim.predict_proba(x_test)
    tt_probs = tt.predict_proba(x_test)
    probs = alpha*mlr_probs + beta*sim_probs + gama*tt_probs
    print 'test evaluation: '
    evaluate(y_test, probs, 1)


if __name__ == '__main__':
    freeze_support()

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str, default="")
    argparser.add_argument("--tags_file", type=str, default="")
    argparser.add_argument("--model_file_mlr", type=str, default="")
    argparser.add_argument("--model_file_tt", type=str, default="")

    argparser.add_argument("--truncate", type=int, default=1)
    argparser.add_argument("--njobs", type=int, default=3)
    argparser.add_argument("--cross_val", type=int, default=0)

    args = argparser.parse_args()
    print args
    main()


