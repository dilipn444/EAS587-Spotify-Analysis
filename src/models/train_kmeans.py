# ============================================================
# train_kmeans.py — Algorithm 4: k-Means Clustering
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Clustering / Unsupervised (in-class)
# Runtime : 2–4 minutes
# ============================================================

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Global settings ──────────────────────────────────────────
from pathlib import Path

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "cleaned_spotify_tracks.csv"

FEATURES = [
    'danceability', 'energy', 'valence', 'acousticness',
    'instrumentalness', 'tempo', 'loudness', 'speechiness'
]

np.random.seed(SEED)
FEATURES       = ['danceability', 'energy', 'valence', 'acousticness',
                  'instrumentalness', 'tempo', 'loudness', 'speechiness']
np.random.seed(SEED)

# ── Train ────────────────────────────────────────────────────
print('=== ALGORITHM 4: k-Means Clustering ===')
print('Runtime estimate: 2–4 minutes (elbow + silhouette sweep)\n')
t0 = time.time()

df_km   = pd.read_csv(PROCESSED_PATH)
X_km    = df_km[FEATURES].dropna()
y_tiers = df_km.loc[X_km.index, 'popularity_tier'].astype(str)

sample_n   = min(20000, len(X_km))
sample_idx = X_km.sample(n=sample_n, random_state=SEED).index
X_sample   = X_km.loc[sample_idx]
y_sample   = y_tiers.loc[sample_idx]

km_scaler = StandardScaler()
X_sc      = km_scaler.fit_transform(X_sample)

k_range    = range(2, 11)
inertias   = []
sil_scores = []

print('Running elbow + silhouette sweep (k=2..10)...')
for k in k_range:
    km_tmp = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km_tmp.fit_predict(X_sc)
    inertias.append(km_tmp.inertia_)
    sil = silhouette_score(X_sc, labels, sample_size=5000, random_state=SEED)
    sil_scores.append(sil)
    print(f'  k={k:2d}  Inertia={km_tmp.inertia_:,.0f}  Silhouette={sil:.4f}')

best_k_km = list(k_range)[int(np.argmax(sil_scores))]
print(f'\n✅ Best k = {best_k_km} (silhouette = {max(sil_scores):.4f})')

# Figure 4A: Elbow + Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_range, inertias, marker='o', color='steelblue', linewidth=2)
ax1.axvline(best_k_km, color='red', linestyle='--', label=f'Best k={best_k_km}')
ax1.set_xlabel('k'); ax1.set_ylabel('Inertia (Within-Cluster SSE)')
ax1.set_title('Algorithm 4 — k-Means: Elbow Method'); ax1.legend()

ax2.plot(k_range, sil_scores, marker='s', color='coral', linewidth=2)
ax2.axvline(best_k_km, color='red', linestyle='--', label=f'Best k={best_k_km}')
ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette Score')
ax2.set_title('Algorithm 4 — k-Means: Silhouette Score vs. k'); ax2.legend()
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo4_kmeans_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

km_final       = KMeans(n_clusters=best_k_km, random_state=SEED, n_init=10)
cluster_labels = km_final.fit_predict(X_sc)
final_sil      = silhouette_score(X_sc, cluster_labels, sample_size=5000, random_state=SEED)
print(f'\nFinal Silhouette Score (k={best_k_km}): {final_sil:.4f}')

# Figure 4B: PCA 2D Cluster Scatter
pca_km = PCA(n_components=2, random_state=SEED)
X_2d   = pca_km.fit_transform(X_sc)

plt.figure(figsize=(10, 7))
scatter    = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab10', alpha=0.4, s=6)
centers_2d = pca_km.transform(km_final.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], marker='X', s=250, c='black', zorder=5, label='Centroids')
plt.colorbar(scatter, label='Cluster ID')
plt.xlabel(f'PC1 ({pca_km.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_km.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title(f'Algorithm 4 — k-Means (k={best_k_km})\nPCA 2D Projection', fontsize=13)
plt.legend(); plt.tight_layout()
plt.savefig(f'{BASE}/results/algo4_kmeans_cluster_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 4C: Tier distribution per cluster
X_sample_copy = X_sample.copy()
X_sample_copy['cluster']         = cluster_labels
X_sample_copy['popularity_tier'] = y_sample.values

tier_dist = X_sample_copy.groupby(['cluster', 'popularity_tier']).size().unstack(fill_value=0)
tier_pct  = tier_dist.div(tier_dist.sum(axis=1), axis=0) * 100
tier_pct.plot(kind='bar', figsize=(9, 5), colormap='Set2', edgecolor='black')
plt.title(f'Algorithm 4 — k-Means (k={best_k_km})\nPopularity Tier Distribution per Cluster', fontsize=13)
plt.xlabel('Cluster'); plt.ylabel('% of Tracks'); plt.xticks(rotation=0)
plt.legend(title='Popularity Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo4_kmeans_tier_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

profile = X_sample_copy.groupby('cluster')[FEATURES].mean().round(3)
print('\nCluster Centroids (original feature scale):')
print(profile.to_string())

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
