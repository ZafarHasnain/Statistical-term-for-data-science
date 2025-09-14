import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Example dataset
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.rand(15) * 10,
    'Feature2': np.random.rand(15) * 20,
    'Feature3': np.random.rand(15) * 5,
    'Feature4': np.random.rand(15) * 15
})

# 2. K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data)

# 3. Cluster profiling (mean values per cluster)
cluster_profile = data.groupby('Cluster').mean()

# 4. Radar Chart Setup
categories = list(cluster_profile.columns)
categories.remove('Cluster') if 'Cluster' in categories else None
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the loop

# 5. Plot radar chart for each cluster
plt.figure(figsize=(8, 8))
for cluster in cluster_profile.index:
    values = cluster_profile.loc[cluster].values.flatten().tolist()
    values += values[:1]  # close the loop
    plt.polar(angles, values, marker='o', label=f'Cluster {cluster}')

plt.xticks(angles[:-1], categories)
plt.title("Cluster Profiling - Radar Chart", size=15, y=1.05)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.show()
