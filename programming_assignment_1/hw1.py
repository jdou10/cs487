# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 15:22:40 2022

@author: lenovo

CS487
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# For question 1, read the value from dataset

TRAIN_URL='iris.csv'

# file parameter
#pd.read_csv(filepath_or_buffer,header,names)    #header=0( the first line as header，header=None means no topic

column_names=['SepalLength','SepalWidth','PetalLength','PetalWidth', 'Class']
df_iris=pd.read_csv(TRAIN_URL,header=None,names=column_names)
#df_iris.head()    #read former data

iris=np.array(df_iris)

fig=plt.figure('Iris Data',figsize=(15,15))

plt.suptitle("Iris Dara Set\n(Blue->Setosa|Red->Versicolor|Green->Virginical)")

##df = pd.read_csv('iris.data', header=None)
r,c = df_iris.shape
##print("Number of rows and columns of the DataFrame:")
print("rows", r, "columns", c)  # 2. get value of rows ands columns

print(df_iris["Class"].unique())  # 3.print out all the distinct value of last column


# For question 4, print out the rows avg,max,min of Iris-setosa

print("Number of rows in Iris-setosa:")
print(df_iris[df_iris["Class"] == "Iris-setosa"].shape[0])

#iris.iloc[:, 0:3].apply.np.mean
#df_iris.apply(['PetalLength'], np.min)

#col_max = df_iris.max(axis=0)
# For question 5, draw a scatter plot

subsetDataFrame = df_iris[df_iris['Class'] == 'Iris-setosa']
x1 = subsetDataFrame['SepalLength']
y1 = subsetDataFrame['SepalWidth']

subsetDataFrame = df_iris[df_iris['Class'] == 'Iris-versicolor']
x2 = subsetDataFrame['SepalLength']
y2 = subsetDataFrame['SepalWidth']

subsetDataFrame = df_iris[df_iris['Class'] == 'Iris-virginica']
x3 = subsetDataFrame['SepalLength']
y3 = subsetDataFrame['SepalWidth']

plt(x1, y1)        # plot x and y using default line style and color
plt(x1, y1, 'bo')  # plot x and y using blue circle markers
plt(y1)           # plot y using x as index array 0..N-1
plt(y1, 'r+')
#plt(x1, y1, 'go--', linewidth=2, markersize=12)
plt(x1, y1, color='green', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt(x2, y2, color='Blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)
plt(x3, y3, color='Red', marker='o', linestyle='dashed',
     linewidth=2, markersize=12)