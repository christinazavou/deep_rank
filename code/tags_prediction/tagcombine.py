import argparse
import os
import pickle
import sys
from datetime import datetime
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


def get_data(df, type_name, labels):
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
        else:
            self.clf = None

    def fit(self, x_train, y_train):
        if self.clf is None:
            self.clf = OneVsRestClassifier(MultinomialNB(), n_jobs=1)
            self.clf.fit(x_train, y_train)
            pickle.dump(self.clf, open(self.model_file, 'wb'), protocol=2)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)


class SimComponent(object):

    def __init__(self):
        self.tf_idf_trans = TfidfTransformer()
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        x_train = self.tf_idf_trans.fit_transform(x_train)
        print 'x_train shape ', x_train.shape
        self.x_train = x_train
        self.y_train = y_train

    def predict_proba(self, x_eval):
        x_eval = self.tf_idf_trans.transform(x_eval)
        y_scores = []
        for x in x_eval:
            similarities = cosine_similarity(self.x_train, x)
            top_50 = np.argsort(similarities, 0)[:50]
            top_50 = top_50.reshape(50)
            top_50_y = self.y_train[top_50, :]
            vote_per_tag = np.sum(top_50_y, 0)
            sum_not_per_tag = np.zeros(len(vote_per_tag))
            score_per_rank = np.zeros(len(vote_per_tag))
            for i, vote in enumerate(vote_per_tag):
                sum_not_per_tag[i] = np.sum(vote_per_tag[:i]) + np.sum(vote_per_tag[(i+1):])
                if vote == 0:
                    score_per_rank[i] = 0
                else:
                    score_per_rank[i] = vote / sum_not_per_tag[i]
            y_scores.append(score_per_rank)
        return np.array(y_scores)


class TagTermComponent(object):

    def __init__(self):
        self.affinity_term_tag = None

    def fit(self, x_train, y_train):
        n_term_tag = np.matmul(x_train.toarray().T, y_train)  # num_of_terms x num_of_tags
        n_tag = np.sum(y_train, 0)
        self.affinity_term_tag = n_term_tag / (n_tag + 1e-9)  # num_of_terms x num_of_tags

    def predict_proba(self, x):
        y_scores = []
        for xi in x:
            x_terms, _ = np.nonzero(xi)
            y_scores.append(1. - np.prod(1. - self.affinity_term_tag[x_terms, :], axis=0))
        return np.array(y_scores)


def evaluate(test_x, test_y, y_scores):
    """"""
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    eval_labels = []
    for label in range(test_y.shape[1]):
        if (test_y[:, label] == np.ones(test_y.shape[0])).any():
            eval_labels.append(label)
    print '\n{} labels out of {} will be evaluated (zero-sampled-labels removed).'.format(len(eval_labels), test_y.shape[1])
    eval_samples = []
    for sample in range(test_y.shape[0]):
        if (test_y[sample, :] == np.ones(test_y.shape[1])).any():
            eval_samples.append(sample)
    print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), test_y.shape[0])
    print type(test_y), test_y.shape
    test_x = test_x[eval_samples, :]
    test_y = test_y[eval_samples, :]
    test_y = test_y[:, eval_labels]
    print test_x.shape, test_x.dtype, type(test_x), test_y.shape, test_y.dtype, type(test_y)
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    y_scores = y_scores[eval_samples, :]
    y_scores = y_scores[:, eval_labels]

    ev = Evaluation(y_scores, None, test_y)

    return (ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10),
            ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10))


def effective_weights(x_train, y_train, x_dev, y_dev):   # , sample_size, performance

    tt = TagTermComponent()
    mlr = MLRComponent(args.model_file)
    sim = SimComponent()
    tt.fit(x_train, y_train)
    mlr.fit(x_train, y_train)
    sim.fit(x_train, y_train)

    tt_probs = tt.predict_proba(x_dev)
    mlr_probs = mlr.predict_proba(x_dev)
    sim_probs = sim.predict_proba(x_dev)

    best_performance = -1
    best_weights = (0, 0, 0)
    for alpha in np.linspace(0, 1, 10):
        for beta in np.linspace(0, 1, 10):
            for gama in np.linspace(0, 1, 10):
                probs = alpha*mlr_probs + beta*sim_probs + gama*tt_probs
                (p1, p3, p5, p10, r1, r3, r5, r10) = evaluate(x_dev, y_dev, probs)
                if r10 > best_performance:
                    best_weights = (alpha, beta, gama)
                    best_performance = r10
                    print 'alpha beta gama ', alpha, beta, gama
                    print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\tR@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\n'.format(
                        p1, p3, p5, p10, r1, r3, r5, r10
                    )
    return best_weights, tt, mlr, sim


def main():
    print 'Starting at: ', str(datetime.now())

    df = read_df(args.df_path)
    df = df.fillna(u'')
    label_tags = pickle.load(open(args.tags_file, 'rb'))
    print '\nloaded {} tags'.format(len(label_tags))

    x_train, y_train = get_data(df, 'train', label_tags)
    x_dev, y_dev = get_data(df, 'dev', label_tags)
    x_test, y_test = get_data(df, 'test', label_tags)
    del df

    if args.test:
        x_train, y_train = x_train[0:1000], y_train[0:1000]

    bow_vec = CountVectorizer(min_df=20, max_df=1.0, ngram_range=(1, 1), binary=True)
    x_train = bow_vec.fit_transform(x_train)
    print 'x_train shape ', x_train.shape

    x_dev = bow_vec.transform(x_dev)
    (alpha, beta, gama), tt, mlr, sim = effective_weights(x_train, y_train, x_dev, y_dev)

    x_test = bow_vec.transform(x_test)
    tt_probs = tt.predict_proba(x_test)
    mlr_probs = mlr.predict_proba(x_test)
    sim_probs = sim.predict_proba(x_test)
    probs = alpha*mlr_probs + beta*sim_probs + gama*tt_probs
    (p1, p3, p5, p10, r1, r3, r5, r10) = evaluate(x_test, y_test, probs)
    print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\tR@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\n'.format(
        p1, p3, p5, p10, r1, r3, r5, r10
    )


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str, default="/home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv")
    argparser.add_argument("--tags_file", type=str, default="/home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p")
    argparser.add_argument("--model_file", type=str, default="/media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/TagCombine/model.p")

    argparser.add_argument("--truncate", type=int, default=1)
    argparser.add_argument("--njobs", type=int, default=2)
    argparser.add_argument("--test", type=int, default=0)

    args = argparser.parse_args()
    print args
    main()

