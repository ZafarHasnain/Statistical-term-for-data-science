import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

# Generate sample data
X, _ = make_blobs(n_samples=20, centers=3, random_state=42)

# Linkage matrices
methods = ['average', 'complete', 'ward']

plt.figure(figsize=(15, 4))
for i, method in enumerate(methods):
    plt.subplot(1, 3, i+1)
    Z = linkage(X, method=method)
    dendrogram(Z)
    plt.title(f"{method.capitalize()} Linkage")

plt.tight_layout()
plt.show()
