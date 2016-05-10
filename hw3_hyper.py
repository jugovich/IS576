from __future__ import print_function
from contextlib import closing

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

print(__doc__)

X, Y = [], []
with closing(open('red_wine_1.txt', 'rb')) as sledata:
    for line in sledata:
        line = line.replace('\r\n', '')
        line = line.split(',')
        X.append(line[:-1])
        Y.append(int(line[-1]))


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'max_depth': [3, 5, 7, None],
                     'criterion': ['gini', 'entropy'],
                     'min_samples_split': [2, 5, 10, 50, 100, 200, 250],
                     'min_samples_leaf': [1, 2, 5, 10, 20, 50, 100]}]

scores = ['precision', 'recall']

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
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and th