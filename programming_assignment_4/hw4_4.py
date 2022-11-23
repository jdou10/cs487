# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 09:46:22 2022

@author: lenovo
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import KernelPCA


lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, 
random_state=1)
tree_model.fit(X_train_lda, y_train)
X_test_lda = lda.transform(X_test_std)
y_pred = tree_model.predict(X_test_lda)
acc = accuracy_score(y_pred, y_test)
print("DT+LDA acc=", acc)



scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = scikit_kpca.fit_transform(X)

X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)
ppn1 = Perceptron(eta0=0.1, random_state=1)
ppn1.fit(X_train, y_train)
y_pred1 = ppn.predict(X_test)
print("accuracy (no kpca) = ", sum(y_test==y_pred1)/y_test.shape[0])
ppn2 = Perceptron(eta0=0.1, random_state=1)
ppn2.fit(X_train_kpca, y_train)
y_pred2 = ppn.predict(X_test_kpca)
print("accuracy (with kpca) ", sum(y_test==y_pred2)/y_test.shape[0])