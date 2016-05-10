from __future__ import print_function
import os
from contextlib import closing
from collections import defaultdict
from numpy import nan
import pylab as pl
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from math import sqrt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from sklearn.feature_extraction import DictVectorizer

attribute_map = dict([(i, 'A%s' % str(i+1)) for i in xrange(16)])

def _calc_error_rate_conf_int(confusion_matrix):
    error = 0
    total = 0
    for i in range(len(confusion_matrix)):
        for ie in range(len(confusion_matrix[i])):
            if i != ie:
                error += confusion_matrix[i][ie]

            total += confusion_matrix[i][ie]

    _error = float(error)/float(total)
    _variance = _error*(1-_error)
    _se = sqrt(_variance/total)
    _upper = _error + (2*_se)
    _lower = _error - (2*_se)
    return [_error,  _lower, _upper]

def missing_report():
    missing = defaultdict(int)
    rows = 0
    _missing_rows = 0

    with closing(open(os.path.join(__file__, '..', 'credit.csv'), 'rb')) as file:
        for line in file:
            row = line.split(',')
            _missing_row = False
            for i, col in enumerate(row):
                if col == '?':
                    missing[i] += 1
                    if not _missing_row:
                        _missing_row = True
                        _missing_rows += 1

            rows += 1

    total = 0

    for i, v in missing.iteritems():
        total += v
        print('%s,%i,%0.2f%%' % (attribute_map[i], v, ((float(v)/float(rows))*100)))

    print ('Total Data,%i,%0.2f%%' % (total, (float(total)/(float(rows)*16))*100))
    print ('Total Rows,%i,%0.2f%%' % (_missing_rows, (float(_missing_rows)/(float(rows)))*100))

def clean_missing():
    """missing "?" replaced with null"""
    data = pd.read_csv(os.path.join(__file__, '..', 'credit.csv'))
    oldNewMap = {'+': 1, '-': 0}
    data['A16'] = data['A16'].map(oldNewMap)
    data = data.replace({'A2': nan}, 28.46)
    data = data.replace({'A14': nan}, 160.00)
    return data
    # """Replace with constant, a missing value may provide additional information. Create boxplot to show chart"""
    # oldNewMap = {nan: '9'}
    # data['A1'] = data['A1'].map(oldNewMap)
    # oldNewMap = {'u': 1, 'y': 2, 'l': 3, 't': 4, nan: 9}
    # data['A4'] = data['A4'].map(oldNewMap)
    # oldNewMap = {'g': 1, 'p': 2, 'gg': 3, nan: 9}
    # data['A5'] = data['A5'].map(oldNewMap)
    # oldNewMap = {'c': 1, 'd': 2, 'cc': 3, 'i': 4, 'j': 5, 'k': 6,
    #              'm': 7, 'r': 8, 'q': 9, 'w': 10, 'x': 11, 'e': 12,
    #              'aa': 13, 'ff': 14, nan: 99}
    # data['A6'] = data['A6'].map(oldNewMap)
    # oldNewMap = {'v': 1, 'h': 2, 'bb': 3, 'j': 4, 'n': 5, 'z': 6,
    #              'dd': 7, 'ff': 8, 'o': 9, nan: 99}
    # data['A7'] = data['A7'].map(oldNewMap)
    # oldNewMap = {'t': 1, 'f': 2}
    # data['A9'] = data['A9'].map(oldNewMap)
    # data['A10'] = data['A10'].map(oldNewMap)
    # data['A12'] = data['A12'].map(oldNewMap)
    # oldNewMap = {'g': 1, 'p': 2, 's': 3, nan: 9}
    # data['A13'] = data['A13'].map(oldNewMap)
    #
    #
    #
    # return data


def pre_process():
    """Function to pre_process the data"""
    pass


def classify(X, Y, test):
    print('BEGIN %0.2f TRAINING -----------------' % test)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'max_depth': [3, 5, 7, 9, None],
                         'min_samples_split': [26, 52, 78, 104],
                         'min_samples_leaf': [13, 26, 39, 52]
                         }]

    scores = ['precision',]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=10, scoring=score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        # for params, mean_score, scores in clf.grid_scores_:
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean_score, scores.std() / 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    y_true, y_pred = y_test, clf.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)

    print('model score on train data data:')
    print(clf.score(X_train, y_train))

    print('model score on test data')
    print(clf.score(X_test, y_test))


    #print('Gini Importance')
    #print(clf.feature_importances_)

    'Classification Report'
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    'Confusion Matrix'
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print(_calc_error_rate_conf_int(cm))

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic, test size = %s' % str(test))
    pl.legend(loc="lower right")
    pl.savefig(os.path.join(__file__, '..', 'roc_%s.png' % str(test).replace('.', '')))
    pl.clf()
    print('DONE. -----------------')

    return _calc_error_rate_conf_int(cm) + [len(y_test), clf.best_params_]

def display_tree(X, Y, vec, size, parent, child, depth):

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=size, random_state=0)

    clf = tree.DecisionTreeClassifier(min_samples_split=parent, min_samples_leaf=child, max_depth=depth)
    clf = clf.fit(X_train, y_train)

    dot_data = StringIO()
    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(os.path.join(__file__, '..', 'tree_%s.pdf' % str(size).replace('.', '')))
    graph.write_jpeg(os.path.join(__file__, '..', 'tree_%s.jpeg' % str(size).replace('.', '')))


def compare_models(e1, e2, n1, n2):
    _q = float((e1+e2))/2.0
    _sr = float(sqrt(_q*(1.0-_q)*((1.0/n1) + (1/n2))))
    return abs(e1-e2) / _sr

def run():
    missing_report()
    data = clean_missing()
    X, Y = [], []
    vec = DictVectorizer(sparse=False)
    _samples =[]

    for line in data.iterrows():
        _samples.append(dict([(attr, line[1][attr]) for attr in ['A%i' % i for i in xrange(1, 16)]]))
        Y.append(line[1]['A16'])

    X = vec.fit_transform(_samples)


    # e1, lo1, u1, l1 = _run_classifier(X, Y, 42, 21, None)
    error = {}
    low = {}
    upper = {}
    length = {}

    for _test in [.34]:# .9, .8, .7, .6, .5, .4, .3, .2, .1]:
        e, l, u, s, parameters = classify(X, Y, _test)
        error[_test] = e
        low[_test] = l
        upper[_test] = u
        length[_test] = s
        display_tree(X, Y, vec, _test, parameters['min_samples_split'], parameters['min_samples_leaf'], parameters['max_depth'])

    # tests = [.333, .9, .8, .7, .6, .5, .4, .3, .2, .1]
    # while tests:
    #     test1 = tests.pop()
    #     for test2 in [.333, .9, .8, .7, .6, .5, .4, .3, .2, .1]:
    #         if test1 == test2:
    #             continue
    #
    #         print(test1, test2)
    #         print(compare_models(error[test1], error[test2], length[test1], length[test2]))




run()