import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Generate sample data
X, _ = make_blobs(n_samples=10, centers=3, random_state=42)

# 2. Compute the linkage matrix
Z = linkage(X, method='ward')  # method can be 'average', 'complete', etc.

# 3. Create dendrogram
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
