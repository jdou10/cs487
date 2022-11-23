from scipy.special import comb
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import operator

'''
This program is testing Adaboost, bagging,
and random forest using digit dataset
'''

digits = load_digits()
X = digits.data
y = digits.target


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.25, random_state=0)
# Use digit dataset test DT
clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
#print(y_pred)
clf.score(X_test, y_test)



print("Following values are tested with Digit Data set")
print('')

# Random Forest
# method
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1,
                                                    stratify=y)

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=8,
                              random_state=1)

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=40, 
                                random_state=1,
                                n_jobs=2)

# prediction

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

forest = forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

forest_train = accuracy_score(y_train, y_train_pred) 
forest_test = accuracy_score(y_test, y_test_pred) 
print('Random forest train/test accuracies %.3f/%.3f'
      % (forest_train, forest_test))
print('')



#--------------------------------------------------------------------------



# # Bagging -- Building an ensemble of classifiers from bootstrap samples
# ## Applying bagging to classify samples



X_train, X_test, y_train, y_test =  train_test_split(X, y, 
                             test_size=0.2, 
                             random_state=1,
                             stratify=y)


tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=450, 
                        max_samples=1.0, 
                        max_features=1.0, 
                        bootstrap=True, 
                        bootstrap_features=False, 
                        n_jobs=2, 
                        random_state=1)


# prediction

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))
print('')

#---------------------------------------------------------------------------



# ## AdaBoost using scikit-learn


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                             test_size=0.2, 
                             random_state=1,
                             stratify=y)



tree = DecisionTreeClassifier(criterion='gini', 
                              max_depth=8,
                              random_state=1)

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=400, 
                         learning_rate=0.1,
                         random_state=1)

#prediction

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))


#----------------------------------------------------------------------------


