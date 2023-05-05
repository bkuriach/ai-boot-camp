# Project Overview
# Implement the k-means algorithm and apply your implementation on the given dataset,
# which contains a set of 2-D points.

# Import Libraries
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import time
print("\nProgram Started :",time.asctime())
# Function to assign data to clusters using minimum euclidean distance to centroids.
# Inout: Data and Centroids
# Output: Assigned Clusters
def assign(data,Centroids):

    EuclideanDistance = np.array([]).reshape(m, 0)
    for k in range(K):
        dist = np.sum((data - Centroids[:, k]) ** 2, axis=1)
        EuclideanDistance = np.c_[EuclideanDistance, dist]
    Clusters = np.argmin(EuclideanDistance, axis=1) + 1

    return(Clusters)

# Function to map clusters and the respective data points
# Input: data and number of clusters
# Output: Map Cluster to Data Points
def map_cluster_data(data, K):
    clusterDataMap = {}
    for k in range(K):
        clusterDataMap[k + 1] = np.array([]).reshape(2, 0)

    for i in range(m):
        clusterDataMap[clusters[i]] = np.c_[clusterDataMap[clusters[i]], data.iloc[i]]

    for k in range(K):
        clusterDataMap[k + 1] = clusterDataMap[k + 1].T

    return(clusterDataMap)

# Function to calculate centroid
# Input: Map with cluster and Data Points and Centroids
# Output: New centroids which are calculated from the data mapping of clusters
def centroid(clusterDataMap,Centroids):

    for k in range(K):
        Centroids[:, k] = np.mean(clusterDataMap[k + 1], axis=0)

    return(Centroids)

# Strategy 1 - Cluster Initialization
# Function to initialize cluster centroids randomly
# Input: Data and Number of Clusters
# Output: Centroids
def initialize_centroids(data, K):
    Centroids = np.array([]).reshape(data.shape[1], 0)
    for i in range(K):
        randIndex = random.randint(0, data.shape[0] - 1)
        Centroids = np.c_[Centroids, data.iloc[randIndex]]
    return(Centroids)

# Strategy 2 - Cluster Initialization
# Function to initialize cluster centroids randomly
# Input: Data and Number of Clusters
# Output: Centroids
def initialize_centroids_strategy2(data, K):
    Centroids = np.array([]).reshape(data.shape[1], 0)
    for i in range(K):
        if i ==0:
            rand = random.randint(0, data.shape[0] - 1)
            Centroids = np.c_[Centroids, data.iloc[rand]]
            # centroidIndexes.append(rand)
            data=data.drop(data.index[rand])
        else:
            centroidMean = np.mean(Centroids,axis=1)
            index = np.argmax(np.sqrt(np.sum((data - centroidMean) ** 2, axis=1)),axis=0)
            # centroidIndexes.append(index)
            Centroids = np.c_[Centroids, data.iloc[index]]
            data = data.drop(data.index[index])
    return(Centroids)

# Read Data
Numpyfile= scipy.io.loadmat("clustering/data/AllSamples.mat")
data = pd.DataFrame(Numpyfile['AllSamples'])
data.columns=['x1','x2']
m = data.shape[0]
n = data.shape[1]

# Initialize Prameters
n_iter = 50

# Initialize plot parameters
color=['red','blue','green','cyan','magenta','grey', 'yellow', 'orange', 'black', 'purple']
labels=['cluster1','cluster2','cluster3','cluster4','cluster5','cluster6', 'cluster7','cluster8','cluster9','cluster10']

#
print("Strategy 1 : First Iteration")
#                  ********* Strategy 1 ************
# Randomly pick the initial centers from the given samples.

# First run with cluster initiation
# Run K-Means with clusters in the range of 2 - 10
WCSS_array=np.array([])
for K in range(2,11):
    Centroids = initialize_centroids(data, K)
    for i in range(n_iter):
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
    wcss = 0
    # Compute Objective functions
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

# Plot the objective function
KMeans_array=np.arange(2,11,1)
plt.figure()
plt.plot(KMeans_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Strategy1 - Run 1: Elbow Chart to identify optimum cluster number')
plt.show()

print("Strategy 1 : Second Iteration")
# Second run with different cluster initiation
# Run K-Means with clusters in the range of 2 - 10
#WCSS --> Within cluster sum of squared errors
WCSS_array=np.array([])
for K in range(2,11):
    Centroids = initialize_centroids(data, K)
    for i in range(n_iter):
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
    wcss = 0
    # Compute Objective functions
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

# Plot the objective function
KMeans_array=np.arange(2,11,1)
plt.figure()
plt.plot(KMeans_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Strategy1 - Run 2: Elbow Chart to identify optimum cluster number')
plt.show()

print("Strategy 2 : First Iteration")
#            ********** Strategy 2 ************
# Strategy 2: pick the first center randomly; for the i-th center (i>1),
# choose a sample (among all possible samples) such that the average distance of this
# chosen one to all previous (i-1) centers is maximal.

# First run with cluster initiation
# Run K-Means with clusters in the range of 2 - 10
WCSS_array=np.array([])
for K in range(2,11):
    Centroids = initialize_centroids_strategy2(data, K)
    for i in range(n_iter):
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
    wcss = 0
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

# Plot the objective function: Strategy 2 - First initialization
KMeans_array=np.arange(2,11,1)
plt.figure()
plt.plot(KMeans_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Strategy2 - Run 1: Elbow Chart to identify optimum cluster number')
plt.show()

print("Strategy 2 : Second Iteration")
# Second run with different cluster initiation
# Run K-Means with clusters in the range of 2 - 10
WCSS_array=np.array([])
for K in range(2,11):
    Centroids = initialize_centroids_strategy2(data, K)
    for i in range(n_iter):
        clusters = assign(data, Centroids)
        clusterDataMap = map_cluster_data(data, K)
        Centroids = centroid(clusterDataMap, Centroids)
    wcss = 0
    for k in range(K):
        wcss += np.sum((clusterDataMap[k + 1] - Centroids[:, k]) ** 2)
    WCSS_array = np.append(WCSS_array, wcss)

# Plot the objective function: Strategy 2 - Second initialization
KMeans_array=np.arange(2,11,1)
plt.figure()
plt.plot(KMeans_array,WCSS_array)
plt.xlabel('Number of Clusters')
plt.ylabel('within-cluster sums of squares (WCSS)')
plt.title('Strategy2 - Run 2: Elbow Chart to identify optimum cluster number')
plt.show()

print("\nProgram Ended :",time.asctime())