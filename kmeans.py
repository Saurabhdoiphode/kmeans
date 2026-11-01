import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("loan_data.csv")

# Display first 5 rows
print(data.head())

# Select features
X = data[['fico', 'dti']]

# Elbow Method

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

data['Cluster'] = kmeans.fit_predict(X)

# Display clustered data
print(data.head())

# Plot clusters
plt.scatter(
    data['fico'],
    data['dti'],
    c=data['Cluster'],
    s=50
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title("Loan Customer Segmentation using K-Means")
plt.xlabel("FICO Score")
plt.ylabel("Debt-to-Income Ratio")
plt.show()