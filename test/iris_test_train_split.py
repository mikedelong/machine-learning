#!/usr/bin/python

""" this example borrows heavily from the example
    shown on the sklearn documentation:

    http://scikit-learn.org/stable/modules/cross_validation.html

"""

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
features = iris.data
labels = iris.target

###############################################################
### YOUR CODE HERE
###############################################################

import sklearn.cross_validation

features_train, features_test, labels_train, labels_test = \
    sklearn.cross_validation.train_test_split(features, labels, random_state=0, test_size=0.4)
### import the relevant code and make your train/test split
### name the output datasets features_train, features_test,
### labels_train, and labels_test

### set the random_state to 0 and the test_size to 0.4 so
### we can exactly check your result




###############################################################

clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)

print clf.score(features_test, labels_test)


##############################################################
def submitAcc():
    return clf.score(features_test, labels_test)