import numpy as np
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

# Read the CSV file
dataset_path = "C:\\Users\\AIZAZ\\Desktop\\Datasets\\karate.csv"
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
    node1_train, node1_test, node2_train, node2_test = train_test_split(node1, node2, test_size=0.2)
    
    # Create the feature matrix
    features = np.column_stack((node1_train, node2_train))
    
    # Compute the distance matrix
    distance_matrix = pairwise_distances(features)
    
    # Apply PAM clustering
    initial_medoid_indexes = np.random.choice(range(len(features)), n_clusters, replace=False)
    pam_instance = kmedoids(distance_matrix, initial_medoid_indexes, data_type='distance_matrix')
    pam_instance.process()
    train_labels = pam_instance.predict(distance_matrix)
    
    # Compute the Xie-Beni index
    cluster_centers = pam_instance.get_medoids()
    intra_cluster_distances = np.sum(distance_matrix[cluster_centers, :][:, cluster_centers])
    inter_cluster_distances = np.sum(distance_matrix)
    xie_beni_index = intra_cluster_distances / inter_cluster_distances
    xie_beni_indices.append(xie_beni_index)

average_xie_beni = np.mean(xie_beni_indices)

print("Average Xie-Beni index over", n_iterations, "iterations:", average_xie_beni)
