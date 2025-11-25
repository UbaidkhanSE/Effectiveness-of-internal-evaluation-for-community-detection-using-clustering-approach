import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import AffinityPropagation

# Load the dataset
dataset_path = "C:\\Users\\AIZAZ\\Desktop\\Datasets\\political_Books.csv"
data = pd.read_csv(dataset_path)

# Create a label encoder to convert categorical values to numeric labels
label_encoder = LabelEncoder()

# Encode categorical columns into numeric labels
data['node1_label'] = label_encoder.fit_transform(data['node1'])
data['node2_label'] = label_encoder.fit_transform(data['node2'])

# Prepare the input data for Affinity Propagation algorithm
input_data = data[['node1_label', 'node2_label']]

# Initialize a list to store Xie-Beni index values
xie_beni_indices = []

# Perform clustering and calculate Xie-Beni index 100 times
for _ in range(100):
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(input_data, test_size=0.2, random_state=42)

    # Apply Affinity Propagation clustering on the training data
    affinity_propagation = AffinityPropagation(random_state=42)
    affinity_propagation.fit(train_data)

    # Predict clusters for the test data
    test_clusters = affinity_propagation.predict(test_data)

    # Calculate the distances between the test data points and the cluster exemplars
    distances = pairwise_distances(test_data, affinity_propagation.cluster_centers_)

    # Calculate the Xie-Beni index
    xie_beni_index = (distances.min(axis=1) ** 2).sum() / (len(affinity_propagation.cluster_centers_) * distances.sum())

    # Add the Xie-Beni index to the list
    xie_beni_indices.append(xie_beni_index)

# Calculate the average Xie-Beni index
average_xie_beni_index = sum(xie_beni_indices) / len(xie_beni_indices)
print("Average Xie-Beni Index:", average_xie_beni_index)
