import argparse
import os
import pickle
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from evaluation import Evaluation
from sklearn.model_selection import PredefinedSplit
from sklearn.utils import shuffle
from sklearn.svm import SVC


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


def evaluate(test_x, test_y, model):
    """"""
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    eval_samples = []
    for sample in range(test_y.shape[0]):
        if (test_y[sample, :] == np.ones(test_y.shape[1])).any():
            eval_samples.append(sample)
    print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), test_y.shape[0])
    print type(test_y), test_y.shape
    test_x = test_x[eval_samples, :]
    test_y = test_y[eval_samples, :]
    # test_y = test_y[:, eval_labels]
    print test_x.shape, test_x.dtype, type(test_x), test_y.shape, test_y.dtype, type(test_y)
    """------------------------------------------remove ill evaluation-------------------------------------------"""

    y_scores = model.predict_proba(test_x)  # probability for each class
    predictions = model.predict(test_x)  # 1 or 0 for each class

    ev = Evaluation(y_scores, predictions, test_y)
    print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\tR@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\tUBP@5: {}\tUBP@10: {}\tMAP: {}\n'.format(
        ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10),
        ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10), ev.upper_bound(5), ev.upper_bound(10),
        ev.MeanAveragePrecision()
    )

    """------------------------------------------remove ill evaluation-------------------------------------------"""
    print 'outputs before ', y_scores.shape
    eval_labels = []
    for label in range(test_y.shape[1]):
        if (test_y[:, label] == np.ones(test_y.shape[0])).any():
            eval_labels.append(label)
    print '\n{} labels out of {} will be evaluated (zero-sampled-labels removed).'.format(len(eval_labels), test_y.shape[1])
    y_scores, predictions, targets = y_scores[:, eval_labels], predictions[:, eval_labels], test_y[:, eval_labels]
    print 'outputs after ', y_scores.shape
    ev = Evaluation(y_scores, predictions, targets)
    print 'precision recall f1 macro: {}'.format(ev.precision_recall_fscore('macro'))
    print 'precision recall f1 micro: {}'.format(ev.precision_recall_fscore('micro'))


def write_sampled_negatives(df_path, tf_idf_vec, model, labels, pfile, N_neg=100):
    df = read_df(df_path)
    df = df.fillna(u'')
    data_x = (df['title'] + u' ' + df['body_truncated']).values
    data_y = df[labels].values
    ids = df['id'].values

    data_x = tf_idf_vec.transform(data_x)
    print data_x.shape, data_x.dtype, type(data_x), len(ids)

    y_scores = model.predict_proba(data_x)  # probability for each class
    y_scores = np.equal(data_y, 0).astype(np.float32) * y_scores  # keep only scores of negative tags
    ranked_scores = y_scores.argsort()[:, ::-1][:, 0:N_neg]
    sampled_tags = np.array(labels)[ranked_scores]
    print ranked_scores.shape

    samples_dict = {}
    for i, q_id in enumerate(ids):
        samples_dict[q_id] = (ranked_scores[i], sampled_tags[i])
    pickle.dump(samples_dict, open(pfile, 'wb'), 2)


def get_data(df, type_name, labels):
    df = df[df['type'] == type_name]
    if args.truncate:
        return (df['title'] + u' ' + df['body_truncated']).values, df[labels].values
    else:
        return (df['title'] + u' ' + df['body']).values, df[labels].values


def save_model(clf, filename):
    print 'saving...'
    pickle.dump(clf, open(filename, 'wb'), protocol=2)


def load_model(filename):
    print 'loading...'
    return pickle.load(open(filename, 'rb'))


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

    tf_idf_vec = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))
    x_train = tf_idf_vec.fit_transform(x_train)
    x_dev = tf_idf_vec.transform(x_dev)
    x_test = tf_idf_vec.transform(x_test)
    print x_train.shape, x_train.dtype, type(x_train)
    print x_dev.shape, x_dev.dtype, type(x_dev)
    print x_test.shape, x_test.dtype, type(x_test)

    if os.path.isfile(args.model_file):

        tuned_model = load_model(args.model_file)

    else:

        print 'Building the model...'
        exit()

        if args.method == 'logreg':
            clf = LogisticRegression(solver='sag', verbose=10, class_weight='balanced', n_jobs=3)
        elif args.method == 'naivebayes':
            clf = MultinomialNB()
        elif args.method == 'linearsvm':
            raise Exception('any svm is not supported')
            clf = LinearSVC(verbose=10)
        else:
            raise Exception('unknown method')

        if args.cross_val:
            x = scipy.sparse.vstack([x_train, x_dev, x_test])
            y = np.vstack((y_train, y_dev, y_test))
            x, y = shuffle(x, y)
            print 'x y ', x.shape, y.shape
            total = x.shape[0]
            train_total = total * 0.8
            dev_total = train_total * 0.2
            x_dev, y_dev = x[0: int(dev_total)], y[0: int(dev_total)]
            x_train, y_train = x[int(dev_total): int(train_total)], y[int(dev_total): int(train_total)]
            x_test, y_test = x[int(train_total):], y[int(train_total):]
            print 'dev train test ', x_dev.shape, x_train.shape, x_test.shape

            # test_fold = [0 for _ in x_train] + [-1 for _ in x_dev]
            # ps = PredefinedSplit(test_fold)
            tx = scipy.sparse.vstack([x_train, x_dev])
            ty = np.vstack((y_train, y_dev))

            # clf = OneVsRestClassifier(CalibratedClassifierCV(clf, cv=ps), n_jobs=1)
            # clf.fit(tx, ty)
            clf = OneVsRestClassifier(clf)
            clf.fit(tx, ty)

        else:

            # clf = OneVsRestClassifier(clf)
            # clf.fit(x_train, y_train)

            # from sklearn.ensemble import BaggingClassifier
            # n_estimators = 10
            # clf = OneVsRestClassifier(
            #     BaggingClassifier(
            #         SVC(kernel='linear', probability=True, class_weight='auto', verbose=1),
            #         max_samples=1.0 / n_estimators,
            #         n_estimators=n_estimators,
            #         n_jobs=args.njobs,
            #         verbose=1
            #     )
            # )
            clf = OneVsRestClassifier(clf, n_jobs=1,)
            clf.fit(x_train, y_train)

        save_model(clf, args.model_file)
        tuned_model = clf

    if args.samples_file:
        write_sampled_negatives(args.df_path, tf_idf_vec, tuned_model, label_tags, args.samples_file, N_neg=args.n_neg)
    else:
        print 'EVALUATE ON DEV\n'
        evaluate(x_dev, y_dev, tuned_model)  # in case of cross val don't look at dev eval

        print 'EVALUATE ON TEST\n'
        evaluate(x_test, y_test, tuned_model)

    print 'Finished at: ', str(datetime.now())


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str)
    argparser.add_argument("--tags_file", type=str)
    argparser.add_argument("--method", type=str, default='logreg')
    argparser.add_argument("--model_file", type=str)

    argparser.add_argument("--cut_off", type=int, default=5)
    argparser.add_argument("--max_df", type=float, default=0.75)
    argparser.add_argument("--n_grams", type=int, default=1)
    argparser.add_argument("--truncate", type=int, default=1)
    argparser.add_argument("--njobs", type=int, default=3)

    argparser.add_argument("--cross_val", type=int, default=0)
    argparser.add_argument("--samples_file", type=str)
    argparser.add_argument("--n_neg", type=int, default=100)

    args = argparser.parse_args()
    print args, '\n'
    main()

