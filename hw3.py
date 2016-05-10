from contextlib import closing

from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
import pydot


def problem1():
    #create an array for the training samples and sample labels
    X, Y = [], []
    with closing(open('sledata.txt', 'rb')) as sledata:
        for line in sledata:
            line = line.split(' ')
            X.append(line[:-1])
            Y.append(line[-1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    clf = tree.DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=1)
    clf = clf.fit(X_train, y_train)

    print 'ten fold cross-validation results:'
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print scores
    print clf.get_params()
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'model score on test set'
    print clf.score(X_test, y_test)

    print 'Gini Importance'
    print clf.feature_importances_

    'Classification Report for test data'
    #print(classification_report(X_test, y_test))

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("sledata.pdf")
    graph.write_jpeg("lopus.jpeg")


def problem2():
    #create an array for the training samples and sample labels
    X, Y = [], []
    with closing(open('winequality-red.csv', 'rb')) as wine:
        for line in wine:
            line = line.split(',')
            X.append(line[:-1])
            Y.append(line[-1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    clf = tree.DecisionTreeClassifier(min_samples_split=200, min_samples_leaf=20)
    clf = clf.fit(X_train, y_train)

    print 'ten fold cross-validation results:'
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'model score (no train/test split)'
    print clf.score(X_test, y_test)

    print 'Gini Importance'
    print clf.feature_importances_
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("wine1.pdf")
    graph.write_jpeg("wine1.jpeg")

def problem2_new():
    #create an array for the training samples and sample labels
    X, Y, weights = [], [], []
    with closing(open('red_wine_1.txt', 'rb')) as wine:
        for line in wine:
            line = line.replace('\r\n', '')
            line = line.split(',')
            line = [float(i) if '.' in i else int(i) for i in line]

            X.append(line[:-1])
            Y.append(line[-1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    clf = tree.DecisionTreeClassifier(min_samples_split=200, min_samples_leaf=50)
    clf = tree.DecisionTreeClassifier(min_samples_split=200, min_samples_leaf=20)
    clf = clf.fit(X_train, y_train)

    print 'ten fold cross-validation results:'
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print 'model score (no train/test split)'
    print clf.score(X_test, y_test)

    print 'Gini Importance'
    print clf.feature_importances_
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("wine2.pdf")
    graph.write_jpeg("wine2.jpeg")


#problem1()
problem2()
#problem2_new()
