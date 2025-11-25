from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

from sklearn.cluster import MiniBatchKMeans



# Split the adjacency matrix into training and testing sets
train_data, test_data = train_test_split(adj_matrix, test_size= 0.5, random_state=42)

# Initialize the Mini-Batch K-Means model with the number of clusters
kmeans = MiniBatchKMeans(n_clusters=4,batch_size=1024)

# Fit the model to the training data
kmeans.fit(train_data)

# Get the cluster labels for the test data
prediction = kmeans.predict(test_data)


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

