# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:47:31 2022

@author: lenovo
"""
# perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Perceptron (object) :

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
# Fit training data
 
    def fit(self, X, y):
         rgen = np.random.RandomState(self.random_state)
         self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
         self.errors_ = []
         for _ in range(self.n_iter):
             errors = 0
             for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) 
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
             self.errors_.append(errors)
         return self
            
            
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
            
df = pd.read_csv('iris.csv', header=None)
y = df.iloc[0:100, 4].values # select setosa and versicolor
y = np.where(y == 'Iris-setosa', -1, 1) # Convert the class labels to two integer
X = df.iloc[0:100, [0, 2]].values # extract sepal length and petal length
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')


# Adaline
class Adaline (object) :
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
        
        net_input = self.net_input(X)
        output = net_input
        errors = (y - output)
        self.w_[1:] += self.eta * X.T.dot(errors)
        self.w_[0] += self.eta * errors.sum()
        cost = (errors**2).sum() / 2.0
        self.cost_.append(cost)

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('iris.csv', header=None)
y = df.iloc[0:100, 4].values # select setosa and versicolor
y = np.where(y == 'Iris-setosa', -1, 1) # Convert the class labels to two integer
X = df.iloc[0:100, [0, 2]].values # extract sepal length and petal length
ppn = Adaline(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.cost_) + 1), ppn.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
    

# Stochastic Gradient Descent (SGD)
class SGD (object) :
        
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
    
        for i in range(self.n_iter):
            X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(y)
                self.cost_.append(avg_cost)

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    
    
def main():
    datafile = 'iris.csv'
    #datafile = 'abalone.csv'
    classifier = "perceptron"
    df = pd.read_csv(datafile,header=None)
    y = df.iloc[0:100, 4].values # select setosa and versicolor
    y = np.where(y == 'Iris-setosa', -1, 1) # Convert the class labels to two integer
    X = df.iloc[0:100, [0, 2]].values # extract sepal length and petal length
    if classifier == "perceptron":   
        ppn = Perceptron(eta=0.1, n_iter=20)
        ppn.fit(X, y)
        print(ppn.errors_)
        plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
    df = pd.read_csv(datafile, header=None)
    
    if classifier == "adaline":   
        ppn = Adaline(eta=0.1, n_iter=20)
        ppn.fit(X, y)
        print(ppn.errors_)
        plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
    df = pd.read_csv(datafile, header=None)
    
    if classifier == "sgd":   
        ppn = SGD(eta=0.1, n_iter=10)
        ppn.fit(X, y)
        print(ppn.errors_)
        plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
    df = pd.read_csv(datafile, header=None)

if __name__ == "__main__":
    main()
    
    
