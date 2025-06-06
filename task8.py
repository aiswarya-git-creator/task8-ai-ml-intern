import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows of dataset:")
print(df.head())
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], color='darkorange', marker='o')
plt.xlabel("Annual Income (k$)",fontweight='bold')
plt.ylabel("Spending Score (1-100)",fontweight='bold')
plt.title("Customer Data - Before Clustering",fontweight='bold')
plt.suptitle("This plot is used to understand how customers are distributed based on income and spending.", fontsize=10, y=0.95)
plt.show()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss, marker='o', color='navy', linestyle='--')
plt.title("Elbow Method to Find Optimal K",fontweight='bold')
plt.xlabel("Number of Clusters (K)",fontweight='bold')
plt.ylabel("WCSS",fontweight='bold')
plt.suptitle("The “elbow point” (where the decrease in WCSS slows down) suggests the best K to choose. Here, K=5 is optimal.", fontsize=10, y=0.95)
plt.show()
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)
df["Cluster"] = y_kmeans
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c=y_kmeans, cmap='tab10')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c='red', label='Centroids')
ax.set_xlabel("Annual Income (k$)",fontweight='bold')
ax.set_ylabel("Spending Score (1-100)",fontweight='bold')
ax.set_title("K-Means Clustering (K=5)",fontweight='bold')
suptitles = [
    "Each distinct color (like brown, green, grey, blue ,lightblue.) represents a cluster of customers identified by the K-Means algorithm.Since you used K=5, there are 5 clusters, each shown in a different color.",
    "The large green dots are the centroids (the center of each cluster)",
    "This visualizes the result of K-Means clustering, showing how the algorithm grouped similar customers based on income and spending score."
]

bullet_text = "\n".join(f"* {line}" for line in suptitles)
fig.text(0.01, 0.97, bullet_text, fontsize=10, ha='left', va='top', wrap=True)
plt.show()
score = silhouette_score(X, y_kmeans)
print("Silhouette Score:", round(score, 2))
