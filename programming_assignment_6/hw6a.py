from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import time
import sys



# ## using the K-means algorithm offered by scikit-learn library
start1_time = time.time()
#Generate synthetic data
X,y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, 
                  shuffle=True, random_state=0)

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


y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')
plt.scatter(X[100:150, 0], X[100:150, 1],
            color='green', marker='v', label='virginica')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')



plt.grid()
plt.tight_layout()
plt.show()
print("\n")



#K-means code
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300,
            tol=1e-04,random_state=0)

y_km = km.fit_predict(X)
print("Results for K-means:")
print(y_km)
print(km.cluster_centers_)
print('SE = %.3f' %km.inertia_)

# Plot the points in three clusters and the centroids
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='green',
            marker='o', edgecolor='black',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='blue',
            marker='o', edgecolor='black',
            label='cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


print("--------- %s seconds ---------" % (time.time() - start1_time))
print("\n")




# ## using a hierarchical approach offered by SciPy library,
start2_time = time.time()

#generate random points
np.random.seed(123)

X = np.random.random_sample([5,2]) #generate 5*2 random samples which are in the range of [0,1]
X = X*10 #make the values to be in the range of [1 ,10]
print("This is the random points generated: ")
print(X)



#reformat this to a data frame
print("\n")
print("Results for hierarchical approach in SciPy, to Reformat this to a data frame")
features = ['f1', 'f2']
row_labels = ['p0', 'p1', 'p2', 'p3', 'p4']
df=pd.DataFrame(X,columns=features, index=row_labels) 
print(df, "\n")

#  Use SciPy’s submodule to calculate pair-wise point distance
print("Results after calculate pair-wise point distance: ")
print("\n")
pair_wise_dist_condensed_form = pdist(df, metric='euclidean') 
print(pair_wise_dist_condensed_form)
print("\n")
row_dist = pd.DataFrame(squareform(pair_wise_dist_condensed_form),
                        columns=row_labels, index= row_labels)
print(row_dist)
print("\n")

print("Results for Cluster using SciPy function linkage: ")
row_cluster1 = linkage(df.values, method='complete', metric='euclidean')
print(row_cluster1)
print("\n")


row_dendr = dendrogram(row_cluster1, labels = row_labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()


#  retrieve the final clusters using fcluster
print("Results for final clusters: ")
max_d = 2.5
clusters1 = fcluster(row_cluster1, max_d, criterion='distance')
print("Cluster 1 is: ", clusters1)
k=2
clusters2 = fcluster(row_cluster1, k, criterion='maxclust')
print("Cluster 2 is: ", clusters2)
print("--------- %s seconds --------" % (time.time() - start2_time))
print("\n")
#


# ## using a hierarchical approach offered by scikit-learn library,
start3_time = time.time()
print("Results for hierachical approach using scikit-learn.")
cluster1 = AgglomerativeClustering(n_clusters=3, 
                             affinity='euclidean', 
                             linkage='complete')
cluster1_labels = cluster1.fit_predict(X)
print('Cluster1 labels: %s' % cluster1_labels)

cluster2 = AgglomerativeClustering(n_clusters=2, 
                             affinity='euclidean', 
                             linkage='complete')
cluster2_labels = cluster2.fit_predict(X)
print('Cluster2 labels: %s' % cluster2_labels)
print("--------- %s seconds ----------" % (time.time() - start3_time))
#



# ## elbow method 
#print('Distortion: %.2f' % km.inertia_)
X, y = make_moons(n_samples=800, noise=0.05, random_state=0)
distortions = []
# Calculate distortions
for i in range(1, 11):
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# Plot distortions for different K
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()
#

