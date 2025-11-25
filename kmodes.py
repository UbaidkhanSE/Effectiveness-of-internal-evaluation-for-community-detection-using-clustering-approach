from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
G = nx.read_edgelist("E:\\Datasets\\Unicode_languages.csv", delimiter=',', create_using=nx.Graph(), nodetype=int)
# Draw the graph using the spring layout
nx.draw_spring(G, with_labels=True)
# Display the graph
plt.show()


# Convert the graph to an adjacency matrix
adj_matrix = nx.to_numpy_matrix(G)
adj_matrix = np.asarray(adj_matrix)
print(adj_matrix)

# Split the adjacency matrix into training and testing sets
train_data, test_data = train_test_split(adj_matrix, test_size= 0.5, random_state=42)


km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
clusters = km.fit(train_data)
prediction = km.predict(test_data)
print(km.cluster_centroids_)
labels = km.labels_



# Create a graph
G = nx.Graph()

# Add nodes to the graph, with the cluster label as the node attribute
for i in range(len(train_data)):
    G.add_node(i, label=labels[i])

# Add edges between nodes that are in the same cluster
for i in range(len(train_data)):
    for j in range(i+1, len(train_data)):
        if labels[i] == labels[j]:
            G.add_edge(i, j)

# Use a spring layout to arrange the nodes on the graph
pos = nx.spring_layout(G)
# Draw the nodes, with the color of the node determined by its cluster label"
nx.draw_networkx_nodes(G, pos, node_color=labels, cmap=plt.get_cmap("jet"))
# Draw the edges
nx.draw_networkx_edges(G, pos)
# Display the graph
plt.show()


# Compute the Silhouette Score
scores = []
for i in range(100):
    score = silhouette_score(test_data, prediction)
    scores.append(score)

average_score = np.mean(scores)
print("The average silhouette score after 100 runs is:", average_score)


# Compute the Davies-Bouldin Inde
scores = []
for i in range(100):
    score = davies_bouldin_score(test_data, prediction)
    scores.append(score)

average_score = np.mean(scores)
print("The average davies_bouldin_score after 100 runs is:", average_score)




