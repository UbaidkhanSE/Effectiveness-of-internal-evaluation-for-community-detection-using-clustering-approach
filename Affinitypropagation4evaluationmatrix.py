import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split

# Load the graph from the file
G = nx.read_edgelist("E:\\Datasets\\Wiktionary_edits_(bo).csv", delimiter=',', create_using=nx.Graph(), nodetype=int)

# Draw the graph
nx.draw_spring(G, with_labels=False)
plt.show()

# Get the number of nodes in the graph
n_samples = len(G.nodes())
print("Number of nodes:", n_samples)

# Convert the graph to an adjacency matrix
adj_matrix = nx.to_numpy_matrix(G)
print("Adjacency matrix:\n", adj_matrix)


# Compute the affinity matrix using the cosine similarity measure
affinity_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        affinity_matrix[i, j] = np.dot(adj_matrix[i], adj_matrix[j].T) / (np.linalg.norm(adj_matrix[i]) * np.linalg.norm(adj_matrix[j]))

# Split the affinity matrix into training and testing sets
train_data, test_data = train_test_split(affinity_matrix, test_size=0.5, random_state=42)

# Train the AffinityPropagation model on the training data
# damping: Damping factor in the affinity propagation algorithm
# Controls the influence of incoming messages on each iteration
# A higher value (e.g., closer to 1) encourages more damping and can help prevent oscillations

ap = AffinityPropagation(damping=0.7, random_state=42).fit(train_data)

# Predict the clusters for the test data"
predictions = ap.predict(test_data[:-1])

# Print the predicted labels and cluster centers
print("Predicted labels:", predictions)
print("Cluster centers:\n", ap.cluster_centers_)

# Compute the Silhouette Score
silhouette_score_val = silhouette_score(test_data[:-1], predictions)
print("Silhouette Score:", silhouette_score_val)

# Compute the Davies-Bouldin Index
davies_bouldin_score_val = davies_bouldin_score(test_data[:-1], predictions)
print("Davies-Bouldin Index:", davies_bouldin_score_val)

# Compute the sum of squared errors
sse = np.sum((test_data[:-1] - ap.cluster_centers_[ap.labels_[:len(test_data)-1]]) ** 2)
print("Sum of squared errors:", sse)
