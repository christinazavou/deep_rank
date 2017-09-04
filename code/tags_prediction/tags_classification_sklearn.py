from utils.statistics import read_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pickle
import sys
import argparse
from sklearn.model_selection import PredefinedSplit
import os
import numpy as np
import pandas as pd
# from sklearn.metrics import coverage_error
# coverage_error(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]]))
# from sklearn.metrics import label_ranking_average_precision_score
# label_ranking_average_precision_score(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]]))
# from sklearn.metrics import label_ranking_loss
# label_ranking_loss(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]]))
# from sklearn.metrics import hamming_loss
# hamming_loss(np.array([[1,0,0,1],[0,1,0,1]]), np.array([[1,0,0,1],[1,1,0,0]]))
# from sklearn.metrics import log_loss
# log_loss(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]]))


# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score
# y_true = [[1,2,3],[1],[2]]
# y_pred = [[1,2,3],[2],[1]]
# m = MultiLabelBinarizer().fit(y_true)
# f1_score(m.transform(y_true),
#          m.transform(y_pred),
#          average='micro')

from sklearn.metrics import precision_recall_fscore_support
# precision_recall_fscore_support(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[1, 1, 1], [1, 0, 0]]),average='macro')
# precision_recall_fscore_support(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[1, 1, 1], [1, 0, 0]]),average='micro')

# from sklearn.metrics import average_precision_score
# average_precision_score(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[1, 1, 1], [1, 0, 0]]),average='micro')


def grid_search(train_x, train_y, dev_x, dev_y, parameters, pipeline):
    tx = np.concatenate((train_x, dev_x))
    ty = np.vstack((train_y, dev_y))
    test_fold = [0 for _ in train_x] + [-1 for _ in dev_x]
    ps = PredefinedSplit(test_fold)
    grid_search_tune = GridSearchCV(pipeline, parameters, n_jobs=args.njobs, verbose=10, cv=ps)
    grid_search_tune.fit(tx, ty)

    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print

    # measuring performance on test set
    print "Applying best classifier on test data:"
    best_clf = grid_search_tune.best_estimator_

    return best_clf


def model_fit(train_x, train_y, dev_x, dev_y, pipeline):
    tx = np.concatenate((train_x, dev_x))
    ty = np.vstack((train_y, dev_y))
    pipeline.fit(tx, ty)
    return pipeline


def evaluate(test_x, test_y, labels, model):
    predictions = model.predict(test_x)
    # print 'MACRO RESULTS\n:', classification_report(test_y, predictions, target_names=labels)
    # print 'MICRO RESULTS\n:', classification_report(test_y, predictions, target_names=labels, sample_weight=)
    precision, recall, f1, support = precision_recall_fscore_support(test_y, predictions)
    results = pd.DataFrame({'tag/label': labels, 'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    print results.head(len(labels))
    print 'MACRO PRECISION RECALL F1: ', precision_recall_fscore_support(test_y, predictions, average='macro')
    print 'MICRO PRECISION RECALL F1: ', precision_recall_fscore_support(test_y, predictions, average='micro')


def get_train_test_dev(df, labels):
    data = {}
    for type_name, group_df in df.groupby('type'):
        data[type_name] = ((group_df['title'] + u' ' + group_df['body']).values, group_df[labels].values)
    return data['train'], data['dev'], data['test']


def save_model(clf, filename):
    print 'saving...'
    pickle.dump(clf, open(filename, 'wb'))


def load_model(filename):
    print 'loading...'
    return pickle.load(open(filename, 'rb'))


def main():

    df = read_df(args.df_path)
    df = df.fillna(u'')
    label_tags = pickle.load(open(args.tags_file, 'rb'))

    train_data, dev_data, test_data = get_train_test_dev(df, label_tags)

    x_train, y_train = train_data
    x_dev, y_dev = dev_data
    x_test, y_test = test_data

    if args.test:
        x_train, y_train = x_train[0:100], y_train[0:100]
        x_dev, y_dev = x_dev[0:100], y_dev[0:100]
        x_test, y_test = x_test[0:100], y_test[0:100]

    if args.method == 'logreg':
        clf = OneVsRestClassifier(LogisticRegression(solver='sag', verbose=10), n_jobs=1)
        if args.predefined:
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))
        else:
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df)
            parameters = {
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                "clf__estimator__C": [0.01, 0.1, 1],
                "clf__estimator__class_weight": ['balanced', None],
            }
    elif args.method == 'svm':
        clf = OneVsRestClassifier(LinearSVC(verbose=10), n_jobs=1)
        if args.predefined:
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))
        else:
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df)
            parameters = {
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                "clf__estimator__C": [0.01, 0.1, 1],
                "clf__estimator__class_weight": ['balanced', None],
            }
    elif args.method == 'randforest':
        if args.predefined:
            clf = OneVsRestClassifier(RandomForestClassifier(verbose=10, criterion="entropy"), n_jobs=1)
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))
        else:
            clf = OneVsRestClassifier(RandomForestClassifier(verbose=10), n_jobs=1)
            tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df)
            parameters = {
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                # "clf__estimator__criterion": ["gini", "entropy"],
                "clf__estimator__criterion": ["entropy"],
                # "clf__estimator__n_estimators": [10, 50, 100],
                "clf__estimator__n_estimators": [10, 100],
                # "clf__estimator__class_weight": ['balanced', None]
            }
    else:
        raise Exception('unknown method')

    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', clf),
    ])

    if os.path.isfile(args.model_file):
        tuned_model = load_model(args.model_file)
    else:
        if args.predefined:
            tuned_model = model_fit(x_train, y_train, x_dev, y_dev, pipeline)
        else:
            tuned_model = grid_search(x_train, y_train, x_dev, y_dev, parameters, pipeline)
        save_model(tuned_model, args.model_file)

    print 'EVALUATE ON TEST\n'
    evaluate(x_test, y_test, label_tags, tuned_model)
    print 'EVALUATE ON DEV\n'
    evaluate(x_dev, y_dev, label_tags, tuned_model)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str)

    argparser.add_argument("--cut_off", type=int, default=5)
    argparser.add_argument("--max_df", type=float, default=0.75)
    argparser.add_argument("--n_grams", type=int, default=3)

    # argparser.add_argument("--max_seq_len", type=int, default=-1)
    # argparser.add_argument("--batch_size", type=int, default=40)

    argparser.add_argument("--method", type=str, default='logreg')
    argparser.add_argument("--model_file", type=str)
    argparser.add_argument("--tags_file", type=str)

    argparser.add_argument("--njobs", type=int, default=2)
    argparser.add_argument("--test", type=bool, default=False)
    argparser.add_argument("--predefined", type=bool, default=False)

    args = argparser.parse_args()
    print args
    print
    main()

