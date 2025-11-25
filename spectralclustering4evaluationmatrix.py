import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split


G = nx.read_edgelist("E:\\Datasets\\Wikiquote_edits_(fo).csv", delimiter=',', create_using=nx.Graph(), nodetype=int)
nx.draw_spring(G, with_labels=False)
plt.show()

n_samples = len(G.nodes())
print("n_sample", n_samples)

# Convert the graph to an adjacency matrix
adj_matrix = nx.to_numpy_matrix(G)
print(adj_matrix)
print("Adjcancy matrix size")
row, col = adj_matrix.shape
print(row, col)

import warnings
warnings.filterwarnings('ignore')

# Split the adjacency matrix into training and testing sets
train_data, test_data = train_test_split(adj_matrix, test_size=0.5, random_state=42)
row, col = train_data.shape
print("training size", row, col)
row, col = test_data.shape
print("testing size: ", row, col)

# affinity: Measure of similarity between samples
# 'rbf' stands for radial basis function kernel, which measures the similarity using the Gaussian kernel
# gamma: Parameter for the Gaussian kernel
# Controls the width of the kernel and influences the impact of each sample's neighbors on the clustering
# n_init: Number of times the algorithm will be run with different initializations
# The final clustering result will be based on the run that yields the best objective function value
# assign_labels: Strategy for assigning labels in the spectral clustering
# 'kmeans' assigns labels using K-means algorithm after eigenvector decomposition

# Train the spectral clustering model on the training data
sc = SpectralClustering(n_clusters=2, affinity='rbf', gamma=1.0, n_init=100, assign_labels='kmeans', random_state=42).fit(train_data)
labels = sc.labels_
print("labels: ", labels)

# Predict the clusters for the test data
predictions = sc.fit_predict(test_data)
print("prediction", predictions)


print("\nEvaluation matrixs\n")

# Compute the Silhouette Score
scores = []
for i in range(100):
    score = silhouette_score(test_data, predictions)
    scores.append(score)
average_score = np.mean(scores)
print("The average silhouette score after 100 runs is:", average_score)

# Compute the Davies-Bouldin Index
scores = []
for i in range(100):
    score = davies_bouldin_score(test_data, predictions)
    scores.append(score)
average_score = np.mean(scores)
print("The average Davies-Bouldin score after 100 runs is:", average_score)


# Compute the sum of square error
sse = 0
for i in range(len(predictions)):
    centroid = np.mean(test_data[predictions == predictions[i]], axis=0)
    sse += np.sum(np.square(test_data[i] - centroid))
average_sse = sse / len(predictions)
print("The average sum of square error is:", average_sse)




