from utils.statistics import read_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import argparse
from sklearn.model_selection import PredefinedSplit
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, f_classif
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.svm import SVC


def plot_learning_curve(estimator, filename, X, y, ylim=None, cv=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # todo: how to use? It doesnt fit the estimator, but gives plots

    print 'making plot ...'
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    print 'train_sizes: ', train_sizes
    print 'train_scores: ', train_scores
    print 'test_scores: ', test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.savefig(filename)


def train_with_plot(train_x, train_y, dev_x, dev_y, estimator, filename):
    # todo: how to use? It doesnt fit the estimator, but gives plots

    # todo: use train and dev cross eval
    tx = scipy.sparse.vstack([train_x, dev_x])
    ty = np.vstack((train_y, dev_y))
    test_fold = [0 for _ in train_x] + [-1 for _ in dev_x]
    ps = PredefinedSplit(test_fold)
    plot_learning_curve(estimator, filename, tx, ty, ylim=None, cv=ps, train_sizes=np.linspace(.1, 1.0, 5))
    return estimator


def grid_search(train_x, train_y, dev_x, dev_y, parameters, pipeline):
    print 'available params: ', pipeline.get_params().keys()

    tx = scipy.sparse.vstack([train_x, dev_x])
    ty = np.vstack((train_y, dev_y))
    test_fold = [0 for _ in train_x] + [-1 for _ in dev_x]
    ps = PredefinedSplit(test_fold)

    # grid_search_tune = GridSearchCV(pipeline, parameters, n_jobs=args.njobs, verbose=10, cv=ps)
    grid_search_tune = GridSearchCV(pipeline, parameters, n_jobs=args.njobs, verbose=10, cv=3)
    # print grid_search_tune.estimator.get_params().keys()

    grid_search_tune.fit(tx, ty)

    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print
    print "Scores of cv:"
    print grid_search_tune.cv_results_
    print
    best_clf = grid_search_tune.best_estimator_

    return best_clf


def parameter_tuning(train_x, train_y, dev_x, dev_y, method='svm'):
    f1scores = []
    best_f1 = -1.
    best_model = None

    if method == 'svm':
        C = [0.05, 0.1, 0.5, 1.0]
        for c in C:
            print 'svm with c=', c
            clf = OneVsRestClassifier(SVC(verbose=5, C=c), n_jobs=1)
            clf.fit(train_x, train_y)
            f1scores.append(precision_recall_fscore_support(dev_y, clf.predict(dev_x), average='micro')[2])
            print 'f1 score ', f1scores[-1]
            if f1scores[-1] > best_f1:
                best_model = clf
                best_f1 = f1scores[-1]
                save_model(best_model, args.model_file)
    elif method == 'randforest':
        # N = [2, 5, 8, 12, 15, 18]
        MSL = [2, 6, 18, 46]
        MD = [3, 6, 9, 15, 21]
        # for n in N:
        # for msl in MSL:
        for md in MD:
            # print 'random forest with n=', n
            # print 'random forest with msl=', msl
            print 'random forest with md=', md
            clf = OneVsRestClassifier(
                # RandomForestClassifier(verbose=5, n_estimators=n, n_jobs=2, random_state=100, criterion="entropy"),
                # RandomForestClassifier(verbose=5, min_samples_leaf=msl, n_jobs=2, random_state=100, criterion="entropy"),
                RandomForestClassifier(verbose=5, max_depth=md, n_jobs=2, random_state=100, criterion="entropy"),
                n_jobs=1
            )
            clf.fit(train_x, train_y)
            f1scores.append(precision_recall_fscore_support(dev_y, clf.predict(dev_x), average='micro')[2])
            print 'f1 score ', f1scores[-1]
            if f1scores[-1] > best_f1:
                best_model = clf
                best_f1 = f1scores[-1]
                save_model(best_model, args.model_file)
    elif method == 'logreg':
        C = [0.05, 0.1, 0.5, 1.0]
        for c in C:
            print 'logistic regression with c=', c
            clf = OneVsRestClassifier(LogisticRegression(verbose=5, C=c, solver='sag', n_jobs=2), n_jobs=1)
            clf.fit(train_x, train_y)
            f1scores.append(precision_recall_fscore_support(dev_y, clf.predict(dev_x), average='micro')[2])
            print 'f1 score ', f1scores[-1]
            if f1scores[-1] > best_f1:
                best_model = clf
                best_f1 = f1scores[-1]
                save_model(best_model, args.model_file)

    print 'f1scores ', f1scores
    return best_model


def model_fit(train_x, train_y, dev_x, dev_y, clf):
    # tx = np.concatenate((train_x, dev_x))
    # ty = np.vstack((train_y, dev_y))
    # pipeline.fit(tx, ty)
    clf.fit(train_x, train_y)
    return clf


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


def main():

    df = read_df(args.df_path)
    df = df.fillna(u'')
    label_tags = pickle.load(open(args.tags_file, 'rb'))

    train_data, dev_data, test_data = get_train_test_dev(df, label_tags)

    x_train, y_train = train_data
    x_dev, y_dev = dev_data
    x_test, y_test = test_data

    print len(x_train), len(x_dev), len(x_test)

    if args.test:
        x_train, y_train = x_train[0:1000], y_train[0:1000]
        x_dev, y_dev = x_dev[0:1000], y_dev[0:1000]
        x_test, y_test = x_test[0:1000], y_test[0:1000]

    tfidf = TfidfVectorizer(min_df=args.cut_off, max_df=args.max_df, ngram_range=(1, args.n_grams))
    x_train = tfidf.fit_transform(x_train)
    if args.kselect:
        features = feature_selection(x_train, y_train, K=args.kselect, chi2_method=args.chi2)
    else:
        features = range(x_train.shape[1])
    x_train = x_train[:, features]
    x_dev = tfidf.transform(x_dev)[:, features]
    x_test = tfidf.transform(x_test)[:, features]

    print x_train.shape, x_dev.shape, x_test.shape

    parameters = {}

    if args.method == 'logreg':
        clf = OneVsRestClassifier(LogisticRegression(solver='sag', verbose=10), n_jobs=1)
        parameters.update({
            "clf__estimator__C": [0.01, 0.1, 1],
            "clf__estimator__class_weight": ['balanced', None],
        })
    elif args.method == 'svm':
        clf = OneVsRestClassifier(SVC(verbose=10), n_jobs=1)
        parameters.update({
            "clf__estimator__kernel": ['linear', 'poly', 'rbf'],
            "clf__estimator__class_weight": ['balanced', None],
            "clf__estimator__gamma": [0.01, 0.1, 1, 10, 100],
            "clf__estimator__C": [0.01, 0.1, 1, 10, 100],
        })
    elif args.method == 'randforest':
        if args.predefined:
            clf = RandomForestClassifier(verbose=10, criterion="entropy", n_estimators=args.n, min_samples_leaf=args.msl)
        else:
            clf = RandomForestClassifier(verbose=10)
            parameters.update({
                "clf__criterion": ["gini", "entropy"],
                "clf__n_estimators": [1, 5, 10, 15],
                "clf__class_weight": ['balanced', None],
                "clf__max_depth": [5, 10, 15],
                "clf__min_samples_leaf": [5, 10, 25, 50]
            })
    else:
        raise Exception('unknown method')

    pipeline = Pipeline([
        ('clf', clf)
    ])

    if os.path.isfile(args.model_file):
        tuned_model = load_model(args.model_file)
    else:
        if args.predefined:
            tuned_model = model_fit(x_train, y_train, x_dev, y_dev, clf)
            save_model(tuned_model, args.model_file)
        elif args.param_tune:
            tuned_model = parameter_tuning(x_train, y_train, x_dev, y_dev, args.method)
        else:
            tuned_model = grid_search(x_train, y_train, x_dev, y_dev, parameters, pipeline)
            save_model(tuned_model, args.model_file)

    print 'EVALUATE ON TEST\n'
    evaluate(x_test, y_test, label_tags, tuned_model)
    print 'EVALUATE ON DEV\n'
    evaluate(x_dev, y_dev, label_tags, tuned_model)
    # print 'EVALUATE ON TRAIN\n'
    # evaluate(x_train[0:1000], y_train[0:1000], label_tags, tuned_model)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--df_path", type=str)

    argparser.add_argument("--cut_off", type=int, default=5)
    argparser.add_argument("--max_df", type=float, default=0.75)
    argparser.add_argument("--n_grams", type=int, default=3)

    # argparser.add_argument("--max_seq_len", type=int, default=-1)

    argparser.add_argument("--method", type=str, default='logreg')
    argparser.add_argument("--model_file", type=str)
    argparser.add_argument("--tags_file", type=str)

    argparser.add_argument("--njobs", type=int, default=3)
    argparser.add_argument("--test", type=bool, default=False)
    argparser.add_argument("--predefined", type=bool, default=False)
    argparser.add_argument("--param_tune", type=bool, default=False)
    argparser.add_argument("--kselect", type=int, default=0)

    argparser.add_argument("--msl", type=int, default=1)
    argparser.add_argument("--n", type=int, default=10)
    argparser.add_argument("--chi2", type=bool, default=True)

    args = argparser.parse_args()
    print args
    print
    main()

