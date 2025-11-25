import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import train_test_split
from kmodes.kmodes import KModes
import numpy as np
import community


num_iterations = 50
ami_scores = []

for _ in range(num_iterations):
    # Step 1: Load the dataset
    data = pd.read_csv("C:/Users/AIZAZ/Desktop/Datasets/kangro.csv")

    # Step 2: Create an undirected graph from the dataset
    G = nx.from_pandas_edgelist(data, "node1", "node2")

    # Step 3: Convert the graph to an adjacency matrix
    adj_matrix = nx.to_numpy_matrix(G)

    # Step 4: Split the adjacency matrix into training and testing sets
    train_data, test_data = train_test_split(adj_matrix, test_size=0.5, random_state=42)

    # Step 5: Train the KModes model on the training data
    kmodes = KModes(n_clusters=2, init='Huang', n_init=14, verbose=0, random_state=42).fit(train_data)
    kmodes_labels_train = kmodes.labels_

    # Step 6: Apply KModes model on the entire graph to get labels for all nodes
    kmodes_labels_all = kmodes.predict(adj_matrix)

    # Step 7: Apply Louvain algorithm on the entire graph
    louvain_partition = community.best_partition(G)
    louvain_labels = [louvain_partition[node] for node in G.nodes()]

    # Step 8: Use label matching to find the best correspondence between training labels and Louvain labels
    best_matching = {}
    for kmodes_label in set(kmodes_labels_train):
        kmodes_indices = [i for i, label in enumerate(kmodes_labels_all) if label == kmodes_label]
        best_louvain_label = max(set(louvain_labels), key=lambda x: len(set(kmodes_indices) & set([i for i, label in enumerate(louvain_labels) if label == x])))
        best_matching[kmodes_label] = best_louvain_label

    # Step 9: Map the Louvain labels of the testing data using the best matching from training labels
    adjusted_louvain_labels_test = [best_matching.get(label, -1) for label in louvain_labels]

    # Step 10: Calculate AMI score
    ami_score = adjusted_mutual_info_score(kmodes_labels_all, adjusted_louvain_labels_test)
    ami_scores.append(ami_score)

average_ami_score = np.mean(ami_scores)
print("Average AMI score after", num_iterations, "iterations:", average_ami_score)
