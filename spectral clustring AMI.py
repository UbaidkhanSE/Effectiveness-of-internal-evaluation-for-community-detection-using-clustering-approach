import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
import numpy as np
import community

num_iterations = 100
ami_scores = []

for _ in range(num_iterations):
    # Step 1: Load the dataset
    data = pd.read_csv("C:/Users/AIZAZ/Desktop/Datasets/iceland.csv")

    # Step 2: Create an undirected graph from the dataset
    G = nx.from_pandas_edgelist(data, "node1", "node2")

    # Step 3: Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_matrix(G)

    # Step 4: Split the adjacency matrix into training and testing sets
    train_data, test_data = train_test_split(adj_matrix, test_size=0.4, random_state=42)

    # Step 5: Train the Spectral Clustering model on the training data
    spectral_clustering = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=42).fit(train_data)
    spectral_labels_train = spectral_clustering.labels_

    # Step 6: Apply Spectral Clustering model on the entire graph to get labels for all nodes
    spectral_labels_all = spectral_clustering.fit_predict(adj_matrix)

    # Step 7: Apply Louvain algorithm on the entire graph
    louvain_partition = community.best_partition(G)
    louvain_labels = [louvain_partition[node] for node in G.nodes()]

    # Step 8: Use label matching to find the best correspondence between training labels and Louvain labels
    best_matching = {}
    for spectral_label in set(spectral_labels_train):
        spectral_indices = [i for i, label in enumerate(spectral_labels_all) if label == spectral_label]
        best_louvain_label = max(set(louvain_labels), key=lambda x: len(set(spectral_indices) & set([i for i, label in enumerate(louvain_labels) if label == x])))
        best_matching[spectral_label] = best_louvain_label

    # Step 9: Map the Louvain labels of the testing data using the best matching from training labels
    adjusted_louvain_labels_test = [best_matching.get(label, -1) for label in louvain_labels]

    # Step 10: Calculate AMI score
    ami_score = adjusted_mutual_info_score(spectral_labels_all, adjusted_louvain_labels_test)
    ami_scores.append(ami_score)

average_ami_score = np.mean(ami_scores)
print("Average AMI score after", num_iterations, "iterations:", average_ami_score)
