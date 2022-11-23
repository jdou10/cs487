# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:15:43 2022

@author: lenovo
"""


from sklearn.linear_model import Perceptron
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import time




# perceptron
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
t0 = time.time()
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
ppn.predict(X_test_std)
t1 = time.time()
total = t1-t0
predicted_labels = ppn.predict(X_test_std)
print(accuracy_score(predicted_labels,y_test))
print("Time to run perceptron", total, "millisecond")



# svm
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
t0 = time.time()
#svm = SVC(kernel='rbf', random_state=1, gamma='scale', decision_function_shape='ovr')
svm = SVC(kernel='linear', random_state=1, decision_function_shape='ovr')
svm.fit(X_train_std, y_train)
svm.predict(X_test_std)
t1 = time.time()
total1 = t1-t0
predicted_labels = svm.predict(X_test_std)
print(accuracy_score(predicted_labels,y_test))
print("Time to run support vector machine", total1, "millisecond")



# decision tree
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
t0 = time.time()
dt = DecisionTreeClassifier(criterion='gini', max_depth=8, random_state=1)
dt.fit(X_train_std, y_train)
predicted_labels = dt.predict(X_test_std)
t1 = time.time()
total2 = t1-t0
print(accuracy_score(predicted_labels,y_test))
print("Time to run decision tree", total2, "millisecond")



