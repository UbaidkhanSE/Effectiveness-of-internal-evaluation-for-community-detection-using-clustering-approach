import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

from sklearn.model_selection import train_test_split

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster import cluster_visualizer_multidim
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample

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

# Train the PAM model on the training data
# train_data: Input data for k-medoids clustering
# It should be provided as a list of data points or a distance matrix
# initial_index_medoids: Indices of the initial medoids
# Specifies the initial medoids for the clustering algorithm
# In this case, the medoids are selected at index 0 and 2 of the train_data
# data_type: Type of input data
# 'distance_matrix' indicates that the input data is provided as a distance matrix

pam_instance = kmedoids(train_data.tolist(), initial_index_medoids=[0, 2], data_type='distance_matrix')
pam_instance.process()
medoid_indexes = pam_instance.get_medoids()
labels = pam_instance.predict(train_data.tolist())
print("labels: ", labels)

# Predict the clusters for the test data
test_labels = pam_instance.predict(test_data.tolist())
print("test labels", test_labels)

# Compute the Silhouette Score
scores = []
for i in range(100):
    score = silhouette_score(test_data, test_labels)
    scores.append(score)
average_score = np.mean(scores)
print("The average silhouette score after 100 runs is:", average_score)

# Compute the Davies-Bouldin Index
scores = []
for i in range(100):
    score = davies_bouldin_score(test_data, test_labels)
    scores.append(score)
average_score = np.mean(scores)
print("The average Davies-Bouldin score after 100 runs is:", average_score)

true_labels = np.zeros(len(test_data))
true_labels[0:25] = 1


# Compute the sum of square error
sse = 0
for i in range(len(test_labels)):
    centroid = np.array(train_data[medoid_indexes[test_labels[i]]]).reshape(-1)
    sse += np.sum(np.square(test_data[i] - centroid))
average_sse = sse / len(test_labels)
print("The average sum of square error is:", average_sse)
