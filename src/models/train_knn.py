# ============================================================
# train_knn.py — Algorithm 2: k-Nearest Neighbors
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Classification (in-class)
# Runtime : 3–6 minutes (CV sweep)
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)

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

# ── Helper ───────────────────────────────────────────────────
def load_features(scale=False):
    df = pd.read_csv(PROCESSED_PATH)
    X  = df[FEATURES].dropna()
    y  = df.loc[X.index, 'popularity_tier'].astype(str)
    sc = None
    if scale:
        sc = StandardScaler()
        X  = pd.DataFrame(sc.fit_transform(X), columns=FEATURES, index=X.index)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test, sc

# ── Train ────────────────────────────────────────────────────
print('=== ALGORITHM 2: k-Nearest Neighbors ===')
print('Runtime estimate: 3–6 minutes (CV sweep + prediction)\n')
t0 = time.time()

# k-NN is distance-based — scaling is required
X_train, X_test, y_train, y_test, _ = load_features(scale=True)

k_range   = list(range(3, 22, 2))
cv_scores = []
print('Sweeping k values with 5-fold CV...')
for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    scores  = cross_val_score(knn_tmp, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores.append(scores.mean())
    print(f'  k={k:2d}  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')

best_k = k_range[int(np.argmax(cv_scores))]
print(f'\n✅ Best k = {best_k} (CV accuracy = {max(cv_scores):.4f})')

# Figure 2A: k vs CV Accuracy
plt.figure(figsize=(8, 4))
plt.plot(k_range, cv_scores, marker='o', color='teal', linewidth=2)
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('5-Fold CV Accuracy')
plt.title('Algorithm 2 — k-NN: CV Accuracy vs. k', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo2_knn_k_selection.png', dpi=150, bbox_inches='tight')
plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('--- Classification Report ---')
print(classification_report(y_test, y_pred, zero_division=0))

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Accuracy   : {acc:.4f}')
print(f'Weighted F1: {f1:.4f}')

# Figure 2B: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Greens', colorbar=True)
ax.set_title(f'Algorithm 2 — k-NN (k={best_k})\nConfusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo2_knn_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
