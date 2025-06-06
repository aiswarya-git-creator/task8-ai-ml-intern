# Objective
Perform unsupervised learning using the K-Means clustering algorithm to segment customers based on their annual income and spending score.

# Tools Used
* Python
* Pandas (for data handling)
* Matplotlib (for visualization)
* Scikit-learn (for K-Means and evaluation metrics)

# Dataset
Mall Customers Dataset

# Steps and Description
# Load Dataset
  * Loaded the dataset into a Pandas DataFrame.
  * Displayed the first 5 rows to understand the data structure.
# Select Features
  * Selected two features for clustering:
  * Annual Income (k$)
  * Spending Score (1-100)
# Visualize Raw Data
  * Created a scatter plot showing customer distribution by income and spending.
  * This helped visualize natural groupings before clustering.
# Elbow Method to Find Optimal K
  * Calculated Within-Cluster Sum of Squares (WCSS) for K values from 1 to 10.
  * Plotted the WCSS values.
  * The "elbow" in the plot at K=5 suggests that 5 clusters is the optimal number.
# Apply K-Means Clustering
  * Applied K-Means with K=5.
  * Predicted cluster labels for each customer.
# Visualize Clusters
  * Plotted the clusters with different colors.
  * Marked cluster centroids in red.
  * Added bullet-point explanations as a caption inside the plot.
# Evaluate Clustering
  * Calculated Silhouette Score to evaluate clustering quality.
  * Silhouette Score ranges from -1 to 1; higher values indicate better-defined clusters.

# outputs
* Dataset Preview: Displayed the first 5 rows of the dataset to understand the data.
* Scatter plot showing raw customer data before clustering.
* Elbow method plot to find the optimal number of clusters (K=5).
* Cluster visualization plot with color-coded clusters and centroids.
* Silhouette Score: Printed the Silhouette Score value to evaluate clustering quality (e.g., 0.55).

