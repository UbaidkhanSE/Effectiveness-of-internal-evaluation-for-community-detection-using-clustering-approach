import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
import community  # For Louvain algorithm
from sklearn.model_selection import train_test_split
import numpy as np


num_iterations = 100
ami_scores = []

for _ in range(num_iterations):
    # Step 1: Load the dataset
    data = pd.read_csv("C:/Users/AIZAZ/Desktop/Datasets/unicode_languages.csv")

    # Step 2: Create an undirected graph from the dataset
    G = nx.from_pandas_edgelist(data, "node1", "node2")

    # Step 3: Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_matrix(G)

    # Step 4: Split the adjacency matrix into training and testing sets
    train_data, test_data = train_test_split(adj_matrix, test_size=0.5, random_state=42)

    # Step 5: Train the K-means model on the training data
    kmeans = KMeans(n_clusters=2, n_init=14, max_iter=300, random_state=42).fit(train_data)
    kmeans_labels_train = kmeans.labels_

    # Step 6: Apply K-means model on the entire graph to get labels for all nodes
    kmeans_labels_all = kmeans.predict(adj_matrix)

    # Step 7: Apply Louvain algorithm on the entire graph
    louvain_partition = community.best_partition(G)
    louvain_labels = [louvain_partition[node] for node in G.nodes()]

    # Step 8: Use label matching to find the best correspondence between training labels and Louvain labels
    best_matching = {}
    for kmeans_label in set(kmeans_labels_train):
        kmeans_indices = [i for i, label in enumerate(kmeans_labels_all) if label == kmeans_label]
        best_louvain_label = max(set(louvain_labels), key=lambda x: len(set(kmeans_indices) & set([i for i, label in enumerate(louvain_labels) if label == x])))
        best_matching[kmeans_label] = best_louvain_label

    # Step 9: Map the Louvain labels of the testing data using the best matching from training labels
    adjusted_louvain_labels_test = [best_matching.get(label, -1) for label in louvain_labels]

    # Step 10: Calculate AMI score
    ami_score = adjusted_mutual_info_score(kmeans_labels_all, adjusted_louvain_labels_test)
    ami_scores.append(ami_score)

average_ami_score = np.mean(ami_scores)
print("Average AMI score after", num_iterations, "iterations:", average_ami_score)
