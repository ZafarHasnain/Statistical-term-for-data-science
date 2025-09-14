from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Create sample data with ground truth labels
X, y_true = make_blobs(n_samples=200, centers=3, random_state=42)

# 2. Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)

# 3. Compute ARI
ari_score = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {ari_score:.4f}")
