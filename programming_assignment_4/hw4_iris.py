# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:48:44 2022

@author: lenovo
"""

import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve, auc


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


start1_time = time.time() #import time

# read iris dataset
df = pd.read_csv("iris.data", header=None) 

print(df.head() ) # prints first 5 rows
print("------------------------------") 
print(df.tail() ) # prints last 5 rows

# Preprocess the df
print("df shape", df.shape)
print("\n")

# drop last column 
X = df.iloc[:, :-1] # feature
y = df.iloc[:, -1] # target

print("X \n", X) 
print("y \n ", y) 


# encode the target/y column

print('\t\tAfter label encoding') 
print( np.unique(le.fit_transform(y) ) )
print('\n\n\n')

#y = np.where(y == 'Iris-setosa', -1, 1)
print("y", y)
# extract sepal length and petal length
#X = df.iloc[0:150, [0, 2]].values
#print("X", X)  
#plt.scatter(X[:50, 0], X[:50, 1],
#            color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1],
#            color='blue', marker='x', label='versicolor')
#plt.scatter(X[100:150, 0], X[100:150, 1],
#            color='green', marker='v', label='virginica')

#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')

#plt.grid()
#plt.tight_layout()
#plt.show()
print("\n")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                         shuffle= True,
                                         random_state=0)
print("y_train", y_train)

# Standardizing the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#calculating eigen values
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

#plotting code
#plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
#        label='individual explained variance')
#plt.step(range(1, 14), cum_var_exp, where='mid',
#         label='cumulative explained variance')
#plt.ylabel('Explained variance ratio')
#plt.xlabel('Principal component index')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.show()


# STEP 1: Standardize the d−dimensional data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# STEP 2: Construct the covariance matrix
#print(X_train_std.T.shape)
#cov_mat = np.cov(X_train_std.T)

# STEP 3: Decompose the covariance matrix into its eigenvectors and eigenvalues
#eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
'''
# Select k eigenvectors which correspond to the k largest eigenvalues
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

print(eigen_pairs[0])
print(eigen_pairs[1])
print(eigen_pairs[0][1])

#Construct a projection matrix W from the top k eigenvectors
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)
'''


#X_train_std[0].dot(w)
#Transform the d−dimensional input dataset X using the projection matrix W
# to obtain the new k−dimensional feature subspace
#X_train_pca = X_train_std.dot(w)
#print("1st original instance: ", X_train_std[0])
#print("Instance in PC space: ", X_train_pca[0]) 
#print("Variance in PC1 = %.2f" %np.var(X_train_pca[:,0]))
#print("Variance in PC2 = %.2f" %np.var(X_train_pca[:,1]))


#using scikit-learn library to conduct PCA
pca = PCA()
#X_train_pca = pca.fit_transform(X_train_std)
#pca.explained_variance_ratio_

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


print('\t\t\tX_train PCA:\n', X_train_pca) 



tree_model = DecisionTreeClassifier(criterion='gini', 
                    max_depth=4, random_state=1)
tree_model.fit(X_train_pca, y_train)

y_pred = tree_model.predict(X_test_pca)
acc = accuracy_score(y_pred, y_test)
print("DT+PCA acc=", acc)

# precision, recall, and f1 score testing
#print("y_test", y_test, "y_pred", y_pred)
#print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,
#                                          average=None))
#print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
#print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

#print("--------- %s seconds ---------" % (time.time() - start1_time))
#print("\n")
#完

print( precision_score(y_test, y_pred), average='macro' )

print(np.unique(y_pred) ) 



import sys
sys.exit(1) 


# ## using the Linear Discriminant Analysis method offered by scikit-learn library
start2_time = time.time() # import time

np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[label - 1]))


# Compute the within-class scatter matrix:

d = 4 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          # sum class scatter matrices

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

print("y_train", y_train)
# Better: covariance matrix since classes are not equally distributed:
print('Class label distribution: %s' 
      % np.bincount(y_train)[1:])

d = 13  # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0],
                                                     S_W.shape[1]))


# Compute the between-class scatter matrix:
mean_overall = np.mean(X_train_std, axis=0)
d = 13  # number of features
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    mean_overall = mean_overall.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))



# ## Selecting linear discriminants for the new feature subspace

# Solve the generalized eigenvalue problem for the matrix $S_W^{-1}S_B$:
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])


tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(range(1, 14), discr, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, 14), cum_discr, where='mid',
         label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()


w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)



# ## LDA via scikit-learn
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, 
                                    random_state=1)
tree_model.fit(X_train_lda, y_train)

X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
y_pred = tree_model.predict(X_test_lda)
acc = accuracy_score(y_pred, y_test)
print("DT+LDA acc=", acc)

# precision, recall, and f1 score testing
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average=None))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

print("--------- %s seconds ---------" % (time.time() - start2_time))
print("\n")
#完



# ## Kernel principal component analysis in scikit-learn
start3_time = time.time() # import time

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

# precision, recall, and f1 score testing
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average=None))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

print("--------- %s seconds ---------" % (time.time() - start3_time))
print("\n")
#完
