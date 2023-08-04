from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample dataset of customer data with shopping-related features
# Replace this with your own dataset
# Format: [feature1, feature2, ..., featureN]
dataset = [
    [50, 20],    # Customer 1
    [35, 45],    # Customer 2
    [20, 10],    # Customer 3
    [15, 5],     # Customer 4
    # Add more data here if you have a larger dataset
]

# Create a K-Means clustering model with the desired number of clusters
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the model on the dataset
kmeans.fit(dataset)

# Function to take input from the user for a new customer
def get_new_customer_input(num_features):
    new_customer_features = []
    for i in range(num_features):
        feature_value = float(input(f"Enter feature {i+1}: "))
        new_customer_features.append(feature_value)
    return new_customer_features

# Get input for a new customer
new_customer = get_new_customer_input(num_features=len(dataset[0]))

# Scale the new customer's features (optional but recommended for K-Means)
scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)
new_customer_scaled = scaler.transform([new_customer])

# Use the trained K-Means model to assign the new customer to a cluster
predicted_cluster = kmeans.predict(new_customer_scaled)

print(f"The new customer belongs to cluster {predicted_cluster[0]}")
