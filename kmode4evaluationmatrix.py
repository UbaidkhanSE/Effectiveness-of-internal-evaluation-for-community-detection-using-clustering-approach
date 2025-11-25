import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split


G = nx.read_edgelist("E:\\Datasets\\Wiktionary_edits_(bo).csv", delimiter=',', create_using=nx.Graph(), nodetype=int)
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

# Train the K-modes model on the training data
#huang : It helps to select initial cluster centroids effectively.
#n_init:  parameter represents the number of times the k-modes algorithm will be run with different centroid seeds.
#verbose: refers to the level of detail or amount of information that is displayed during the clustering process.
#random_state: is used to seed the random number generator, ensuring reproducibility of results.
kmodes = KModes(n_clusters=2, init='Huang', n_init=14, verbose=1, random_state=42).fit(train_data)
centroid = kmodes.cluster_centroids_
labels = kmodes.labels_
print("labels: ", labels)

# Predict the clusters for the test data
predictions = kmodes.predict(test_data)
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

# Compute the average sum of square error
sse = 0
for i in range(len(predictions)):
    centroid = kmodes.cluster_centroids_[predictions[i]]
    sse += np.sum(np.square(test_data[i] - centroid))
avg_sse = sse / len(predictions)
print("The average sum of square error is:", avg_sse)