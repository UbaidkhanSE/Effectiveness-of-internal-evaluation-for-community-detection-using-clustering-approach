import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Read the CSV file
dataset_path = "C:\\Users\\AIZAZ\\Desktop\\Datasets\\Americanfootball.csv"
df = pd.read_csv(dataset_path)

# Extract relevant features or columns from the dataset
node1 = df['node1'].values
node2 = df['node2'].values

# Determine the unique nodes
unique_nodes = np.unique(np.concatenate((node1, node2)))

n_iterations = 100
n_clusters = 2
xie_beni_indices = []

for _ in range(n_iterations):
    # Split the data into training and testing sets
    node1_train, node1_test, node2_train, node2_test = train_test_split(node1, node2, test_size=0.5)
    
    # Create the adjacency matrix
    n_nodes = len(unique_nodes)
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    node_index_map = {node: i for i, node in enumerate(unique_nodes)}
    for n1, n2 in zip(node1_train, node2_train):
        i, j = node_index_map[n1], node_index_map[n2]
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1  # Assuming undirected graph
    
    # Apply spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    train_labels = sc.fit_predict(adjacency_matrix)
    
    # Compute the Xie-Beni index
    cluster_centers, _ = pairwise_distances_argmin_min(adjacency_matrix, adjacency_matrix)
    intra_cluster_distances = np.sum(adjacency_matrix[cluster_centers, :])
    inter_cluster_distances = np.sum(pairwise_distances(cluster_centers.reshape(-1, 1), metric='euclidean'))
    xie_beni_index = intra_cluster_distances / inter_cluster_distances
    xie_beni_indices.append(xie_beni_index)

average_xie_beni = np.mean(xie_beni_indices)

print("Average Xie-Beni index over", n_iterations, "iterations:", average_xie_beni)
