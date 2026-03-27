# ============================================================
# train_pca.py — Algorithm 6: PCA (Principal Component Analysis)
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Dimensionality Reduction (outside class)
# Source  : Jolliffe, I. T. (2002). Principal Component Analysis.
#           Springer. https://doi.org/10.1007/b98835
# Runtime : < 30 seconds
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
from sklearn.decomposition import PCA

# ── Global settings ──────────────────────────────────────────
SEED           = 42
BASE           = 'project_repo'
PROCESSED_PATH = f'{BASE}/data/processed/cleaned_spotify_tracks.csv'
FEATURES       = ['danceability', 'energy', 'valence', 'acousticness',
                  'instrumentalness', 'tempo', 'loudness', 'speechiness']
np.random.seed(SEED)

# ── Train ────────────────────────────────────────────────────
print('=== ALGORITHM 6: PCA — Principal Component Analysis (Outside Algorithm) ===')
print('Source: Jolliffe, I.T. (2002). Principal Component Analysis. Springer.')
print('Runtime estimate: < 30 seconds\n')
t0 = time.time()

df_pca = pd.read_csv(PROCESSED_PATH)
X_pca  = df_pca[FEATURES].dropna()
y_pca  = df_pca.loc[X_pca.index, 'popularity_tier'].astype(str)

pca_scaler = StandardScaler()
X_pca_sc   = pca_scaler.fit_transform(X_pca)

# Full PCA — find how many components explain 80% variance
pca_full = PCA(random_state=SEED)
pca_full.fit(X_pca_sc)

cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_80   = int(np.searchsorted(cumvar, 80)) + 1
print(f'Components needed to explain 80% variance: {n_80}')
for i, (ev, cv) in enumerate(zip(pca_full.explained_variance_ratio_, cumvar)):
    print(f'  PC{i+1}: {ev*100:.1f}%  (cumulative: {cv:.1f}%)')

# Figure 6A: Scree + Cumulative Variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(range(1, len(pca_full.explained_variance_ratio_)+1),
        pca_full.explained_variance_ratio_*100, color='steelblue', edgecolor='black')
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Variance Explained (%)')
ax1.set_title('Algorithm 6 — PCA: Scree Plot')

ax2.plot(range(1, len(cumvar)+1), cumvar, marker='o', color='purple', linewidth=2)
ax2.axhline(80, color='red', linestyle='--', label='80% threshold')
ax2.axvline(n_80, color='orange', linestyle='--', label=f'{n_80} components')
ax2.set_xlabel('Number of Components'); ax2.set_ylabel('Cumulative Variance Explained (%)')
ax2.set_title('Algorithm 6 — PCA: Cumulative Explained Variance'); ax2.legend()
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo6_pca_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# 2-component PCA for visualisation
pca_2d = PCA(n_components=2, random_state=SEED)
X_2d   = pca_2d.fit_transform(X_pca_sc)
print(f'\nPC1 explains {pca_2d.explained_variance_ratio_[0]*100:.1f}% of variance')
print(f'PC2 explains {pca_2d.explained_variance_ratio_[1]*100:.1f}% of variance')

# Figure 6B: 2D Scatter by tier
palette = {'Low': '#e74c3c', 'Medium': '#f39c12', 'High': '#2ecc71'}
plt.figure(figsize=(10, 7))
for tier, color in palette.items():
    mask = y_pca == tier
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=tier, alpha=0.3, s=5)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title('Algorithm 6 — PCA\n2D Projection Colored by Popularity Tier', fontsize=13)
plt.legend(markerscale=3); plt.tight_layout()
plt.savefig(f'{BASE}/results/algo6_pca_2d_projection.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 6C: Feature Loadings Heatmap
loadings = pd.DataFrame(pca_2d.components_, columns=FEATURES, index=['PC1', 'PC2'])
plt.figure(figsize=(11, 3))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Algorithm 6 — PCA: Feature Loadings (PC1 & PC2)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo6_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
