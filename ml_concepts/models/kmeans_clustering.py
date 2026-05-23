"""
K-Means Clustering demo
────────────────────────
Covers:
  • K-Means from scratch conceptual walk-through
  • Elbow method for choosing k
  • Silhouette analysis
  • Comparison with Agglomerative hierarchical clustering
  • Cluster quality metrics: inertia, silhouette score, Davies-Bouldin
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from ml_concepts.utils.data_generator import get_clustering_data
from ml_concepts.utils.visualizer import plot_clusters, plot_elbow, plot_model_comparison


def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  K-MEANS CLUSTERING")
    print("=" * 60)

    X, true_labels = get_clustering_data(n_clusters=4)
    X = StandardScaler().fit_transform(X)

    # ── Elbow method ─────────────────────────────────────────────────────
    ks = range(1, 11)
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    path_elbow = f"{plot_dir}/km_elbow.png" if save_plots else None
    plot_elbow(list(ks), inertias, path=path_elbow)

    # ── Silhouette sweep ─────────────────────────────────────────────────
    print("\n  Silhouette scores by k:")
    sil_scores = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        sil_scores.append(s)
        bar = "█" * int(s * 40)
        print(f"    k={k}  sil={s:.4f}  {bar}")

    best_k = np.argmax(sil_scores) + 2
    print(f"\n  Best k by silhouette: {best_k}")

    # ── Final model ──────────────────────────────────────────────────────
    km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=15, random_state=42)
    labels_final = km_final.fit_predict(X)

    print(f"\n  [K-Means  k={best_k}]")
    print(f"    Inertia          : {km_final.inertia_:.4f}")
    print(f"    Silhouette Score : {silhouette_score(X, labels_final):.4f}")
    print(f"    Davies-Bouldin   : {davies_bouldin_score(X, labels_final):.4f}")

    path_cl = f"{plot_dir}/km_clusters.png" if save_plots else None
    plot_clusters(X, labels_final, km_final.cluster_centers_,
                  f"K-Means Clusters (k={best_k})", path=path_cl)

    # ── Compare init strategies ──────────────────────────────────────────
    print("\n  Init strategy comparison (k=4):")
    for init in ("k-means++", "random"):
        km = KMeans(n_clusters=4, init=init, n_init=15, random_state=42)
        lbl = km.fit_predict(X)
        s = silhouette_score(X, lbl)
        print(f"    {init:<12}  inertia={km.inertia_:.2f}  sil={s:.4f}")

    # ── Agglomerative hierarchical comparison ────────────────────────────
    print("\n  Agglomerative clustering (ward linkage, k=4):")
    for linkage in ("ward", "complete", "average"):
        agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
        lbl = agg.fit_predict(X)
        s = silhouette_score(X, lbl)
        db = davies_bouldin_score(X, lbl)
        print(f"    {linkage:<10}  sil={s:.4f}  DB={db:.4f}")

    path_agg = f"{plot_dir}/km_agglomerative.png" if save_plots else None
    agg_final = AgglomerativeClustering(n_clusters=4, linkage="ward")
    plot_clusters(X, agg_final.fit_predict(X), title="Agglomerative (ward, k=4)", path=path_agg)

    # ── Ground truth vs predicted ────────────────────────────────────────
    path_true = f"{plot_dir}/km_true_labels.png" if save_plots else None
    plot_clusters(X, true_labels, title="Ground Truth Labels", path=path_true)

    print("\n  Key takeaways:")
    print("    • k-means++ init dramatically reduces bad local minima vs random init")
    print("    • Elbow + silhouette together give a robust estimate of the right k")
    print("    • Agglomerative clustering is deterministic but scales as O(n² log n)")
    print("    • Always scale features before clustering (distances are scale-sensitive)")
