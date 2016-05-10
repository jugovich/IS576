from contextlib import closing

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
#import pydot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier

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

def compare_models(e1, e2, n):
    _q = float((e1+e2))/2.0
    _sr = sqrt(_q*(1.0-_q)*(2.0/float(n)))
    return abs(e1-e2) / _sr

def _run_k_classifier(X, Y, neighbors):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.333, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf = clf.fit(X_train, y_train)

    print 'model score on train data data:'
    print clf.score(X_train, y_train)
    #print 'ten fold cross-validation results on train data:'
    #scores = cross_val_score(clf, X_train, y_train, cv=10)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'model score on test data'
    print clf.score(X_test, y_test)
    #print 'ten fold cross-validation results on test data:'
    #scores = cross_val_score(clf, X_test, y_test, cv=10)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    'Classification Report'
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    'Confusion Matrix'
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print _calc_error_rate_conf_int(cm)
    return _calc_error_rate_conf_int(cm) + [len(y_test)]

def _run_classifier(X, Y, parent, child, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.333, random_state=0)
    clf = tree.DecisionTreeClassifier(min_samples_split=parent, min_samples_leaf=child, max_depth=max_depth)
    clf = clf.fit(X_train, y_train)

    print 'model score on train data data:'
    print clf.score(X_train, y_train)
    print 'ten fold cross-validation results on train data:'
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'model score on test data'
    print clf.score(X_test, y_test)
    print 'ten fold cross-validation results on test data:'
    scores = cross_val_score(clf, X_test, y_test, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'Gini Importance'
    print clf.feature_importances_

    'Classification Report'
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    'Confusion Matrix'
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print _calc_error_rate_conf_int(cm)
    return _calc_error_rate_conf_int(cm) + [len(y_test)]


    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("cars1.pdf")
    # graph.write_jpeg("cars2.jpeg")

def problem2a():
    #create an array for the training samples and sample labels
    X, Y = [], []

    car = pd.read_csv(r'C:\Users\jugovich-michael\Documents\IS567\car.csv', names=['buying','maint','doors','persons','lug_boot','safety','class'])
    df = pd.DataFrame()

    oldNewMap = {'low': 1, 'med': 2, 'high': 3, 'vhigh':4}
    df['buying'] = car['buying'].map(oldNewMap)

    oldNewMap = {'low': 1, 'med': 2, 'high': 3, 'vhigh':4}
    df['maint'] = car['maint'].map(oldNewMap)

    oldNewMap = {'2': 1, '3': 2, '4': 3, '5more':4}
    df['doors'] = car['doors'].map(oldNewMap)

    oldNewMap = {'2': 1, '4': 2, 'more': 3}
    df['persons'] = car['persons'].map(oldNewMap)

    oldNewMap = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood':4}
    df['class'] = car['class'].map(oldNewMap)

    oldNewMap = {'small': 1, 'med': 2, 'big': 3}
    df['lug_boot'] = car['lug_boot'].map(oldNewMap)

    oldNewMap = {'low': 1, 'med': 2, 'high': 3}
    df['safety'] = car['safety'].map(oldNewMap)

    for line in df.iterrows():
        line = [line[1]['buying'], line[1]['maint'], line[1]['doors'], line[1]['persons'], line[1]['lug_boot'], line[1]['safety'], line[1]['class']]
        X.append(line[:-1])
        Y.append(line[-1])

    e1, lo1, u1, l1 = _run_classifier(X, Y, 42, 21, None)
    print("-"*100)
    e2, lo2, u2, l2 = _run_classifier(X, Y, 42, 21, 5)
    print("-"*100)
    e3, lo3, u3, l1 = _run_classifier(X, Y, 84, 42, None)
    print("-"*100)
    e4, lo4, u4, l1 = _run_classifier(X, Y, 84, 42, 5)
    print("-"*100)
    e5, lo5, u5, l1 = _run_classifier(X, Y, 126, 63, None)
    print("-"*100)
    e6, lo6, u6, l1 = _run_classifier(X, Y, 126, 63, 5)


    print compare_models(e1, e2, l1), compare_models(e1, e3, l1), compare_models(e1, e4, l1), compare_models(e1, e5, l1), compare_models(e1, e6, l1)
    print compare_models(e2, e3, l1), compare_models(e2, e4, l1), compare_models(e2, e5, l1), compare_models(e2, e6, l1)
    print compare_models(e3, e4, l1), compare_models(e3, e5, l1), compare_models(e3, e6, l1)
    print compare_models(e4, e5, l1), compare_models(e4, e6, l1)
    print compare_models(e5, e6, l1)

    k_e1, lo1, u1, l1 = _run_k_classifier(X, Y, 1)
    print("-"*100)
    k_e2, l02, u2, l1 = _run_k_classifier(X, Y, 3)
    print("-"*100)
    k_e3, l03, u3, l1 = _run_k_classifier(X, Y, 5)
    print("-"*100)
    k_e4, l04, u4, l1 = _run_k_classifier(X, Y, 7)
    print("-"*100)

    print compare_models(k_e1, k_e2, l1), compare_models(k_e1, k_e3, l1), compare_models(k_e1, k_e4, l1)
    print compare_models(k_e2, k_e3, l1), compare_models(k_e2, k_e4, l1)
    print compare_models(k_e3, k_e4, l1)


    print compare_models(e1, k_e3, l1)


problem2a()