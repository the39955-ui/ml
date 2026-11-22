from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "iris.csv")
X = df.drop(columns="species")
y = df["species"]  # Used only after clustering to inspect agreement, not to train.
X_scaled = StandardScaler().fit_transform(X)
model = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_scaled)
labels = model.labels_
print("Cluster counts:", pd.Series(labels).value_counts().sort_index().to_dict())
print("Inertia (within-cluster sum of squares):", model.inertia_)
print("Silhouette score:", silhouette_score(X_scaled, labels))
print("Adjusted Rand index vs known species (external check):", adjusted_rand_score(y, labels))
