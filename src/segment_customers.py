from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
FEATURES = ["Age", "AnnualIncome_k", "SpendingScore", "AnnualVisits"]

def main():
    data = pd.read_csv(ROOT / "data" / "customer_data.csv")
    print("Dataset shape:", data.shape)
    print("Missing values:\\n", data.isnull().sum())

    X = data[FEATURES].copy().fillna(data[FEATURES].median())
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow method: lower inertia means tighter clusters.
    ks = range(2, 9)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    plt.figure(figsize=(7, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(OUT / "elbow_method.png", dpi=150)
    plt.close()

    # K=4 is used for this synthetic demonstration dataset.
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(X_scaled)
    print("Silhouette score (K=4):", round(silhouette_score(X_scaled, data["Cluster"]), 3))

    profile = data.groupby("Cluster")[FEATURES].mean().round(2)
    profile["CustomerCount"] = data["Cluster"].value_counts().sort_index()
    profile.to_csv(OUT / "segment_profiles.csv")
    data.to_csv(OUT / "segmented_customers.csv", index=False)
    print("\\nSegment profiles:\\n", profile)

    # PCA reduces the scaled features to two dimensions for visualisation.
    points = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=data["Cluster"], cmap="viridis", alpha=.75)
    plt.title("Customer Segments (PCA View)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUT / "customer_segments.png", dpi=150)
    plt.close()
    print("\\nFinished. Check the outputs/ folder.")

if __name__ == "__main__":
    main()
