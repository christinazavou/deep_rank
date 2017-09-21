import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, label_ranking_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# from tags_prediction.statistics import read_df
# from utils import load_embedding_iterator


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


def load_embedding_iterator(path):
    import gzip
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([float(x) for x in parts[1:]])
                yield word, vals


def grid_search(train_x, train_y, parameters, pipeline, scoring='f1_micro'):

    print 'grid search on ', str(scoring)

    tx = train_x
    ty = train_y

    grid_search_tune = GridSearchCV(
        pipeline, parameters, n_jobs=args.njobs, verbose=10, cv=3, scoring=scoring
    )

    grid_search_tune.fit(tx, ty)

    print
    print("Best parameters:")
    print grid_search_tune.best_params_
    print "Best scores: "
    print grid_search_tune.best_score_
    print
    print "Scores of cv:"
    print grid_search_tune.cv_results_
    print
    best_clf = grid_search_tune.best_estimator_

    return best_clf


def evaluate(test_x, test_y, labels, model):

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

    try:  # because svm has no predict_proba
        y_scores = model.predict_proba(test_x)
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        y_scores = y_scores[:, eval_labels]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        print 'label ranking average precision score: ', label_ranking_average_precision_score(test_y, y_scores)
        print 'coverage error: ', coverage_error(test_y, y_scores)
        print 'label ranking loss: ', label_ranking_loss(test_y, y_scores)
        print
    except:
        pass
    predictions = model.predict(test_x)
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    predictions = predictions[:, eval_labels]
    """------------------------------------------remove ill evaluation-------------------------------------------"""
    precision, recall, f1, support = precision_recall_fscore_support(test_y, predictions)
    results = pd.DataFrame({'tag/label': labels, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    print results.head(len(labels))
    print 'MACRO PRECISION RECALL F1: ', precision_recall_fscore_support(test_y, predictions, average='macro')
    print 'MICRO PRECISION RECALL F1: ', precision_recall_fscore_support(test_y, predictions, average='micro')


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


def feature_selection(x_train, y_train, K=1000, chi2_method=True):
    print 'reducing features from ', x_train.shape[1], ' to ', K
    selected_features = []
    for i in range(y_train.shape[1]):
        if chi2_method:
            selector = SelectKBest(chi2, k='all')
        else:
            selector = SelectKBest(f_classif, k='all')
        selector.fit(x_train, y_train[:, i])
        selected_features.append(list(selector.scores_))
    mean_score_per_feature = np.mean(selected_features, axis=0)
    max_score_per_feature = np.max(selected_features, axis=0)
    idx1 = mean_score_per_feature.argsort()[0:K]
    idx2 = max_score_per_feature.argsort()[0:K]
    print 'common mean-max ', len(set(idx1) & set(idx2))
    return idx1


def make_embedded_representations(embedded_vocab, docs):
    print 'making embedded representation (average of word embeddings)...'
    data_embeddings = []
    for doc in docs:
        text = doc.split(u' ')
        doc_vec = np.zeros(200)
        for w in text:
            doc_vec += embedded_vocab.get(w, embedded_vocab.get('unk'))
        doc_vec /= len(text)
        data_embeddings.append(doc_vec)
    data_embeddings = np.array(data_embeddings)
    return data_embeddings


def transform_data(x, y, vocab=None, embedded=0, given_vocab=0):
    if args.test:
        x, y = x[0:1000], y[0:1000]
        print len(x)

    if embedded:
        assert vocab is not None
        x = make_embedded_representations(vocab, x)
        return x, y, None
    else:
        if given_vocab:
            assert vocab is not None
            print 'using the vocabulary of the embeddings file'
            tf_idf_vec = TfidfVectorizer(vocabulary=vocab)
        else:
            tf_idf_vec = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))

        x = tf_idf_vec.fit_transform(x)
        return x, y, tf_idf_vec


def main():
    print 'Starting at: ', str(datetime.now())

    df = read_df(args.df_path)
    df = df.fillna(u'')
    label_tags = pickle.load(open(args.tags_file, 'rb'))
    if isinstance(label_tags, dict):
        print 'from dict labels to list.'
        label_tags = label_tags.keys()
    print '\nloaded {} tags'.format(len(label_tags))

    x_train, y_train = get_data(df, 'train', label_tags)
    del df

    print len(x_train)

    if args.embedded:
        vocab = {}
        for w, v in load_embedding_iterator(args.emb_file):
            vocab[w] = v
        x_train, y_train, _ = transform_data(x_train, y_train, vocab=vocab, embedded=1)
    else:
        if args.given_vocab:
            vocab = [w for w, v in load_embedding_iterator(args.emb_file)]
            x_train, y_train, tf_idf_vec = transform_data(x_train, y_train, vocab=vocab, given_vocab=1)
        else:
            x_train, y_train, tf_idf_vec = transform_data(x_train, y_train)

        if args.kselect:
            features = feature_selection(x_train, y_train, K=args.kselect, chi2_method=args.chi2)
        else:
            features = range(x_train.shape[1])

        x_train = x_train[:, features]

    if args.norm:
        x_train = normalize(x_train)

    print x_train.shape, x_train.dtype, type(x_train)

    if os.path.isfile(args.model_file):

        tuned_model = load_model(args.model_file)

    else:

        print 'Building the model...'
        parameters = {}

        if args.method == 'logreg':
            clf = OneVsRestClassifier(LogisticRegression(solver='sag', verbose=10), n_jobs=1)
            parameters.update({
                # "estimator__C": [0.01, 0.1, 1],
                # "estimator__class_weight": ['balanced', None],
                "estimator__C": [1],
            })
        elif args.method == 'svm':
            clf = OneVsRestClassifier(SVC(verbose=10), n_jobs=1)
            parameters.update({
                # "estimator__kernel": ['linear', 'poly', 'rbf'],
                "estimator__kernel": ['rbf'],
                # "estimator__class_weight": ['balanced', None],
                "estimator__class_weight": ['balanced'],
                # "estimator__gamma": [0.01, 0.1, 1, 10, 100],
                "estimator__gamma": [1],
                # "estimator__C": [0.01, 0.1, 1, 10, 100],
            })
        elif args.method == 'linearsvm':
            clf = OneVsRestClassifier(LinearSVC(verbose=10), n_jobs=1)
            parameters.update({
                # "estimator__class_weight": ['balanced', None],
                # "estimator__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
                # "estimator__C": [0.1, 1],
                "estimator__C": [1],
            })
        elif args.method == 'randforest':
            clf = OneVsRestClassifier(RandomForestClassifier(verbose=10), n_jobs=1)
            parameters.update({
                "estimator__n_estimators": [10]
            })

            # clf = RandomForestClassifier(verbose=10)
            # parameters.update({
            #     # "criterion": ["gini", "entropy"],
            #     # "n_estimators": [1, 5, 10, 15],
            #     # "class_weight": ['balanced', None],
            #     # "min_samples_leaf": [2, 5, 25, 50]
            #     # "class_weight": ['balanced'],
            #
            #     "max_depth": [50, 60, 70],
            #     "min_samples_leaf": [2, 5, 10]
            # })
        else:
            raise Exception('unknown method')

        print 'parameters to test: ', parameters
        print 'model to build: ', clf

        # _ = grid_search(x_train, y_train, parameters, clf, scoring='precision_macro')
        tuned_model = grid_search(x_train, y_train, parameters, clf, scoring='f1_micro')
        save_model(tuned_model, args.model_file)

    df = read_df(args.df_path)
    df = df.fillna(u'')

    x_dev, y_dev = get_data(df, 'dev', label_tags)
    x_test, y_test = get_data(df, 'test', label_tags)
    print len(x_dev), len(x_test)

    del df

    if args.embedded:
        x_dev = make_embedded_representations(vocab, x_dev)
        x_test = make_embedded_representations(vocab, x_test)
    else:
        x_dev = tf_idf_vec.transform(x_dev)
        x_dev = x_dev[:, features]
        x_test = tf_idf_vec.transform(x_test)
        x_test = x_test[:, features]

    if args.norm:
        x_dev = normalize(x_dev)
        x_test = normalize(x_test)

    print x_dev.shape, x_dev.dtype, type(x_dev)
    print x_test.shape, x_test.dtype, type(x_test)

    print 'EVALUATE ON TEST\n'
    evaluate(x_test, y_test, label_tags, tuned_model)
    print 'EVALUATE ON DEV\n'
    evaluate(x_dev, y_dev, label_tags, tuned_model)

    print 'Finished at: ', str(datetime.now())


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str)
    argparser.add_argument("--tags_file", type=str)
    argparser.add_argument("--emb_file", type=str)
    argparser.add_argument("--method", type=str, default='logreg')
    argparser.add_argument("--model_file", type=str)

    argparser.add_argument("--cut_off", type=int, default=5)
    argparser.add_argument("--max_df", type=float, default=0.75)
    argparser.add_argument("--n_grams", type=int, default=1)
    argparser.add_argument("--truncate", type=int, default=1)
    argparser.add_argument("--embedded", type=int, default=0)
    argparser.add_argument("--given_vocab", type=int, default=0)
    argparser.add_argument("--njobs", type=int, default=2)
    argparser.add_argument("--test", type=int, default=0)

    argparser.add_argument("--msl", type=int, default=1)
    argparser.add_argument("--n", type=int, default=10)

    # NOT TO USE:
    argparser.add_argument("--chi2", type=int, default=1)
    argparser.add_argument("--norm", type=int, default=0)
    argparser.add_argument("--kselect", type=int, default=0)

    args = argparser.parse_args()
    if args.method == 'randforest':
        args.njobs = 1
    print args
    print
    main()

