import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

print("=== WCZYTYWANIE DANYCH ===")

df = pd.read_csv("countries.csv", decimal=",")

print(f"Wymiary danych: {df.shape}")
print(f"\nPierwsze kolumny: {df.columns.tolist()[:10]}")
print("\n=== PREPROCESSING ===")
numeric_features = [
    "Population",
    "Area (sq. mi.)",
    "Pop. Density (per sq. mi.)",
    "GDP ($ per capita)",
    "Literacy (%)",
    "Birthrate",
    "Deathrate",
]

df_clean = df[numeric_features].dropna()
print(f"Liczba krajów po usunięciu braków: {df_clean.shape[0]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(
    f"Wyjaśniona wariancja przez 2 komponenty PCA: {pca.explained_variance_ratio_.sum():.2%}"
)

# ZADANIE 1: K-MEANS Z RÓŻNĄ LICZBĄ SKUPIEŃ
print("\n" + "=" * 60)
print("ZADANIE 1: K-MEANS - WPŁYW LICZBY SKUPIEŃ")
print("=" * 60)

k_values = [2, 3, 4, 5, 6, 7, 8]
kmeans_results = []

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    inertia = kmeans.inertia_

    kmeans_results.append(
        {
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "inertia": inertia,
        }
    )

    print(f"\nK={k}:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
    print(f"  Inertia: {inertia:.2f}")

    axes[idx].scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6
    )
    axes[idx].set_title(f"K-Means (k={k})\nSilhouette: {silhouette:.3f}")
    axes[idx].set_xlabel("PC1")
    axes[idx].set_ylabel("PC2")

axes[7].axis("off")
plt.tight_layout()
plt.savefig("kmeans_comparison.png", dpi=300, bbox_inches="tight")
print("\n✓ Zapisano wykres: kmeans_comparison.png")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_values, [r["inertia"] for r in kmeans_results], "bo-", linewidth=2)
axes[0].set_xlabel("Liczba skupień (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_values, [r["silhouette"] for r in kmeans_results], "go-", linewidth=2)
axes[1].set_xlabel("Liczba skupień (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K")
axes[1].grid(True, alpha=0.3)

axes[2].plot(
    k_values, [r["davies_bouldin"] for r in kmeans_results], "ro-", linewidth=2
)
axes[2].set_xlabel("Liczba skupień (k)")
axes[2].set_ylabel("Davies-Bouldin Index")
axes[2].set_title("Davies-Bouldin Index vs K")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_metrics.png", dpi=300, bbox_inches="tight")
print("✓ Zapisano wykres: kmeans_metrics.png")
plt.show()

# ZADANIE 2: DBSCAN Z RÓŻNYMI PARAMETRAMI
print("\n" + "=" * 60)
print("ZADANIE 2: DBSCAN - TESTOWANIE PARAMETRÓW")
print("=" * 60)

eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [3, 5, 10]
dbscan_results = []

fig, axes = plt.subplots(len(min_samples_values), len(eps_values), figsize=(16, 12))

for i, min_samples in enumerate(min_samples_values):
    for j, eps in enumerate(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        result = {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": n_noise / len(labels),
        }

        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                silhouette = silhouette_score(X_scaled[mask], labels[mask])
                result["silhouette"] = silhouette
            else:
                result["silhouette"] = None
        else:
            result["silhouette"] = None

        dbscan_results.append(result)

        print(f"\neps={eps}, min_samples={min_samples}:")
        print(f"  Liczba skupień: {n_clusters}")
        print(f"  Punkty szumu: {n_noise} ({result['noise_ratio']:.1%})")
        if result["silhouette"]:
            print(f"  Silhouette Score: {result['silhouette']:.4f}")

        axes[i, j].scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6
        )
        axes[i, j].set_title(
            f"eps={eps}, min_s={min_samples}\nClusters={n_clusters}, Noise={n_noise}"
        )
        axes[i, j].set_xlabel("PC1")
        axes[i, j].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("dbscan_comparison.png", dpi=300, bbox_inches="tight")
print("\n✓ Zapisano wykres: dbscan_comparison.png")
plt.show()

# ZADANIE 3: PORÓWNANIE TRZECH METOD
print("\n" + "=" * 60)
print("ZADANIE 3: PORÓWNANIE METOD GRUPOWANIA")
print("=" * 60)

best_k = 4
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

dbscan_final = DBSCAN(eps=1, min_samples=3)
labels_dbscan = dbscan_final.fit_predict(X_scaled)

hierarchical = AgglomerativeClustering(n_clusters=best_k)
labels_hierarchical = hierarchical.fit_predict(X_scaled)

methods = ["K-Means", "DBSCAN", "Hierarchical"]
all_labels = [labels_kmeans, labels_dbscan, labels_hierarchical]
comparison_results = []

for method, labels in zip(methods, all_labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1) if -1 in labels else 0

    result = {"Metoda": method, "Liczba skupień": n_clusters, "Punkty szumu": n_noise}

    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if mask.sum() > 1:
            silhouette = silhouette_score(X_scaled[mask], labels[mask])
            davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
            result["Silhouette Score"] = silhouette
            result["Davies-Bouldin"] = davies_bouldin
        else:
            result["Silhouette Score"] = None
            result["Davies-Bouldin"] = None
    else:
        result["Silhouette Score"] = None
        result["Davies-Bouldin"] = None

    comparison_results.append(result)

comparison_df = pd.DataFrame(comparison_results)
print("\n" + comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (method, labels) in enumerate(zip(methods, all_labels)):
    axes[idx].scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, alpha=0.6
    )
    axes[idx].set_title(
        f"{method}\n({len(set(labels)) - (1 if -1 in labels else 0)} skupień)"
    )
    axes[idx].set_xlabel("PC1")
    axes[idx].set_ylabel("PC2")

plt.tight_layout()
plt.savefig("methods_comparison.png", dpi=300, bbox_inches="tight")
print("\n✓ Zapisano wykres: methods_comparison.png")
plt.show()

# ZADANIE 4: ZASTOSOWANIE DO DANYCH RZECZYWISTYCH - ANALIZA INTERPRETACYJNA
print("\n" + "=" * 60)
print("ZADANIE 4: ANALIZA RZECZYWISTYCH DANYCH KRAJÓW")
print("=" * 60)

df_indexed = df.dropna(subset=numeric_features).copy()
df_indexed["Cluster"] = labels_kmeans

print("\n--- CHARAKTERYSTYKA KLASTRÓW ---")
for cluster in range(best_k):
    cluster_data = df_indexed[df_indexed["Cluster"] == cluster]
    print(f"\n📊 KLASTER {cluster} ({len(cluster_data)} krajów)")
    print(f"Średnie wartości:")
    for feature in numeric_features:
        mean_val = cluster_data[feature].mean()
        print(f"  • {feature}: {mean_val:.2f}")

    if "Country" in df.columns:
        example_countries = cluster_data["Country"].head(5).tolist()
        print(f"  Przykładowe kraje: {', '.join(example_countries)}")

cluster_means = df_indexed.groupby("Cluster")[numeric_features].mean()
cluster_means_normalized = (cluster_means - cluster_means.mean()) / cluster_means.std()

plt.figure(figsize=(12, 8))
sns.heatmap(
    cluster_means_normalized.T,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    cbar_kws={"label": "Znormalizowana wartość"},
)
plt.title("Charakterystyka klastrów krajów (wartości znormalizowane)")
plt.xlabel("Numer klastra")
plt.ylabel("Cechy")
plt.tight_layout()
plt.savefig("cluster_characteristics.png", dpi=300, bbox_inches="tight")
print("\n✓ Zapisano wykres: cluster_characteristics.png")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features_to_plot = [
    "GDP ($ per capita)",
    "Literacy (%)",
    "Birthrate",
    "Pop. Density (per sq. mi.)",
]

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 2, idx % 2]
    df_indexed.boxplot(column=feature, by="Cluster", ax=ax)
    ax.set_title(f"{feature} według klastra")
    ax.set_xlabel("Klaster")
    ax.set_ylabel(feature)

plt.suptitle("Porównanie cech między klastrami", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig("cluster_boxplots.png", dpi=300, bbox_inches="tight")
print("✓ Zapisano wykres: cluster_boxplots.png")
plt.show()

print("\n" + "=" * 60)
print("ANALIZA ZAKOŃCZONA")
print("=" * 60)
