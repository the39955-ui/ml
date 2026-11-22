from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "iris.csv")
X = df.drop(columns="species")
y = df["species"]  # Kept aside for external evaluation only.
X_scaled = StandardScaler().fit_transform(X)
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X_scaled)
print("Cluster counts:", pd.Series(labels).value_counts().sort_index().to_dict())
print("Silhouette score:", silhouette_score(X_scaled, labels))
print("Adjusted Rand index vs known species:", adjusted_rand_score(y, labels))
# A dendrogram for readability: show a reproducible sample of 30 rows.
Z = linkage(X_scaled[:30], method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z)
plt.title("Hierarchical clustering dendrogram (first 30 Iris rows)")
plt.xlabel("Sample index")
plt.ylabel("Ward distance")
plt.tight_layout()
out = ROOT / "outputs" / "hierarchical_dendrogram.png"
plt.savefig(out, dpi=150)
print("Dendrogram saved to:", out)
