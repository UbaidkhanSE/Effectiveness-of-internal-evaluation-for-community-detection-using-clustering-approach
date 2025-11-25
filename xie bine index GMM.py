import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Read the CSV file
dataset_path = "C:\\Users\\AIZAZ\\Desktop\\Datasets\\zebra.csv"
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
    
    # Apply Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(features)
    train_labels = gmm.predict(features)
    
    # Compute the Xie-Beni index
    cluster_centers = gmm.means_
    cluster_covariances = gmm.covariances_
    intra_cluster_distances = np.sum(gmm.score_samples(features))
    inter_cluster_distances = 0
    for i in range(n_clusters):
        inter_cluster_distances += np.sum(gmm.score_samples(cluster_centers[i].reshape(1, -1)))
    xie_beni_index = intra_cluster_distances / inter_cluster_distances
    xie_beni_indices.append(xie_beni_index)

average_xie_beni = np.mean(xie_beni_indices)

print("Average Xie-Beni index over", n_iterations, "iterations:", average_xie_beni)
