import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# Load the dataset
dataset_path = "C:\\Users\\AIZAZ\\Desktop\\Datasets\\kangro.csv"
data = pd.read_csv(dataset_path)

# Create a label encoder to convert node names to numeric labels
label_encoder = LabelEncoder()

# Encode all possible labels in the dataset
all_labels = pd.concat([data['node1'], data['node2']])
label_encoder.fit(all_labels)

# Encode node1 and node2 columns into numeric labels
data['node1_label'] = label_encoder.transform(data['node1'])
data['node2_label'] = label_encoder.transform(data['node2'])

# Create adjacency matrix
adjacency_matrix = pd.crosstab(data['node1_label'], data['node2_label'])

# Specify the number of clusters
num_clusters = 3

# Initialize a list to store Xie-Beni index values
xie_beni_indices = []

# Perform clustering and calculate Xie-Beni index 100 times
for _ in range(100):
    # Split the adjacency matrix into training and testing sets
    train_matrix, test_matrix = train_test_split(adjacency_matrix, test_size=0.2, random_state=42)

    # Apply K-means clustering on the training data
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(train_matrix)

    # Predict clusters for the test data
    test_clusters = kmeans.predict(test_matrix)

    # Calculate the distances between the test data and the cluster centroids
    distances = pairwise_distances(test_matrix, kmeans.cluster_centers_)

    # Calculate the Xie-Beni index
    xie_beni_index = (distances.min(axis=1) ** 2).sum() / (num_clusters * distances.sum())
    
    # Add the Xie-Beni index to the list
    xie_beni_indices.append(xie_beni_index)

# Calculate the average Xie-Beni index
average_xie_beni_index = sum(xie_beni_indices) / len(xie_beni_indices)
print("Average Xie-Beni Index:", average_xie_beni_index)
