from statistics import read_df
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import pickle


NUM_LABELS = 100


def grid_search(train_x, train_y, test_x, test_y, genres, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print
    print("Best parameters set:")
    print grid_search_tune.best_estimator_.steps
    print

    # measuring performance on test set
    print "Applying best classifier on test data:"
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print classification_report(test_y, predictions, target_names=genres)


def get_train_test_dev(df, labels):
    data = {}
    df = df.fillna(u'')
    for type_name, group_df in df.groupby('type'):
        data[type_name] = ((group_df['title'] + u' ' + group_df['body']).values, group_df[labels].values)
    return data['train'], data['dev'], data['test']


def main():
    df = read_df('data.csv')
    NAME = 'all_selected_300.p'

    labels = pickle.load(open(NAME, 'rb'))

    TEST = False
    WAY = 1

    if WAY == 2:
        train_data, dev_data, test_data = get_train_test_dev(df, labels)
        x_train, y_train = train_data
        x_test, y_test = test_data
    else:
        df_x = df['title'] + u' ' + df['body']
        df_y = df[labels]
        data_x = df_x.as_matrix()
        data_y = df_y.as_matrix()

        stratified_split = StratifiedShuffleSplit(n_splits=2, test_size=0.2)
        for train_index, test_index in stratified_split.split(data_x, data_y):
            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]

    print 'x_train ', x_train[0]
    print 'y_train ', y_train[0]
    if TEST:
        x_train, y_train = x_train[0:100], y_train[0:100]
        x_test, y_test = x_test[0:100], y_test[0:100]

    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer(min_df=5, max_df=0.75, ngram_range=(1, 3))),
        ('tfidf', TfidfVectorizer(min_df=5, max_df=0.75, ngram_range=(1, 1))),
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
    ])
    parameters = {
        # "clf__estimator__C": [0.01, 0.1, 1],
        "clf__estimator__C": [1],
        # "clf__estimator__class_weight": ['balanced', None],
    }
    if WAY == 1:
        grid_search(x_train, y_train, x_test, y_test, labels, parameters, pipeline)
    else:
        pipeline.fit(x_train, y_train)
        print
        # measuring performance on test set
        print "Applying classifier on test data:"
        predictions = pipeline.predict(x_test)
        print classification_report(y_test, predictions, target_names=labels)


if __name__ == '__main__':
    main()
