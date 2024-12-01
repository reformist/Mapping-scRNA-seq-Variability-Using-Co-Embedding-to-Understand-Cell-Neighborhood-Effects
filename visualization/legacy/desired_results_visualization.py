import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

# Define parameters for clusters and subclusters
n_clusters = 3  # Number of main clusters
n_subclusters = 2  # Number of subclusters per main cluster
samples_per_subcluster = 100
colors = ['blue', 'red']  # Colors for each subcluster within a cluster

# Generate synthetic data for each main cluster
X, y = [], []
cluster_centers = []

# Adjust main cluster centers to prevent overlap by spacing them out
for i in range(n_clusters):
    main_center = np.array([i * 10, i * 10])  # Centers spaced out by 10 units diagonally
    cluster_centers.append(main_center)
    
    # Generate two subclusters near each main cluster center
    for j in range(n_subclusters):
        X_subcluster, _ = make_blobs(
            n_samples=samples_per_subcluster,
            centers=main_center + np.random.randn(1, 2) * 1.2,  # Keep subclusters near main center
            cluster_std=0.7  # Adjust standard deviation for compactness
        )
        X.append(X_subcluster)
        y.extend([f'Cell Neighborhood {j + 1}'] * samples_per_subcluster)  # Labels for neighborhoods

# Combine all subclusters into single arrays
X = np.vstack(X)

# Map labels to colors
color_map = {
    'Cell Neighborhood 1': colors[1],  # Red for Neighborhood 1
    'Cell Neighborhood 2': colors[0]   # Blue for Neighborhood 2
}
colors_mapped = [color_map[label] for label in y]

# Plot the clusters and subclusters
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=colors_mapped, s=30, edgecolor='k')

# Add labels for neighborhoods and draw circles for each main cluster
for idx, (center, label) in enumerate(zip(cluster_centers, ['Cell Type 1', 'Cell Type 2', 'Cell Type 3'])):
    ellipse = Ellipse(xy=center, width=10, height=10, edgecolor='black', fc='none', lw=2, linestyle='--')
    plt.gca().add_patch(ellipse)
    plt.text(center[0], center[1] + 5.5, label, ha='center', fontsize=12, fontweight='bold')

# Set titles and labels
plt.title("Inter Cell Type Variation Explained by Cell Neighborhood in the scRNA-seq Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Position the legend outside the plot area
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Cell Neighborhood 1 (Red)', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Cell Neighborhood 2 (Blue)', markerfacecolor='blue', markersize=10)
], loc='center left', bbox_to_anchor=(1, 0.5))  # Legend placed outside to the right
plt.xlabel('Genes UMAP 1')
plt.ylabel('Genes UMAP 2')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
# save
plt.savefig('cell_neighborhoods.png')
plt.show()
