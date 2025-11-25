from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import networkx as nx
import numpy as np
G = nx.read_edgelist("E:\\Datasets\\Wikiquote_edits_(fo).csv", delimiter=',', create_using=nx.Graph(), nodetype=int)
# Draw the graph using the spring layout
nx.draw_spring(G, with_labels=True)

# Display the graph
plt.show()
# Convert the graph to an adjacency matrix

adj_matrix = nx.to_numpy_matrix(G)

adj_matrix = np.asarray(adj_matrix)

print(adj_matrix)




# Split the adjacency matrix into training and testing sets
train_data, test_data = train_test_split(adj_matrix, test_size= 0.5,  random_state=42)

# Create an instance of SpectralClustering with the desired number of clusters
sc = SpectralClustering(n_clusters=2)

# Fit the model to the data
sc.fit(train_data)


labels = sc.fit_predict(test_data)

import matplotlib.pyplot as plt
import networkx as nx

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

# Use a circular layout to arrange the nodes on the graph
pos = nx.circular_layout(G)

# Set the node sizes to be proportional to the number of edges
node_sizes = [G.degree(i) * 10 for i in range(len(train_data))]

# Draw the nodes, with the color of the node determined by its cluster label
node_colors = [label / max(labels) for label in labels]  # normalize label values between 0 and 1
cmap = plt.cm.get_cmap("cool")
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap)

# Draw the edges
nx.draw_networkx_edges(G, pos)

# Add a colorbar to show the mapping between node colors and cluster labels
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max(labels)))
sm._A = []
plt.colorbar(sm)

# Display the graph
plt.axis('off')
plt.show()
# Compute the Silhouette Score
scores = []
for i in range(100):
    score = silhouette_score(test_data, labels)
    scores.append(score)

average_score = np.mean(scores)
print("The average silhouette score after 100 runs is:", average_score)


# Compute the Davies-Bouldin Inde
scores = []
for i in range(100):
    score = davies_bouldin_score(test_data, labels)
    scores.append(score)

average_score = np.mean(scores)
print("The average davies_bouldin_score after 100 runs is:", average_score)


