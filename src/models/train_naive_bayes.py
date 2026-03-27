# ============================================================
# train_naive_bayes.py — Algorithm 3: Naive Bayes (Gaussian)
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Classification (in-class)
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)

# ── Global settings ──────────────────────────────────────────
SEED           = 42
BASE           = 'project_repo'
PROCESSED_PATH = f'{BASE}/data/processed/cleaned_spotify_tracks.csv'
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
print('=== ALGORITHM 3: Naive Bayes (Gaussian) ===')
print('Runtime estimate: < 30 seconds\n')
t0 = time.time()

X_train, X_test, y_train, y_test, _ = load_features(scale=False)

# var_smoothing tuning via GridSearchCV
param_grid = {'var_smoothing': np.logspace(-12, -1, 20)}
gs_nb = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs_nb.fit(X_train, y_train)
best_vs = gs_nb.best_params_['var_smoothing']
print(f'Best var_smoothing: {best_vs:.2e}')
print(f'Best CV accuracy  : {gs_nb.best_score_:.4f}')

nb = GaussianNB(var_smoothing=best_vs)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print('--- Classification Report ---')
print(classification_report(y_test, y_pred, zero_division=0))

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Accuracy   : {acc:.4f}')
print(f'Weighted F1: {f1:.4f}')

# Figure 3A: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Purples', colorbar=True)
ax.set_title('Algorithm 3 — Naive Bayes\nConfusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo3_nb_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 3B: Class-conditional feature distributions
df_plot = pd.read_csv(PROCESSED_PATH)
X_all   = df_plot[FEATURES].dropna()
y_all   = df_plot.loc[X_all.index, 'popularity_tier'].astype(str)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes    = axes.flatten()
palette = {'High': '#2ecc71', 'Low': '#e74c3c', 'Medium': '#3498db'}
for i, feat in enumerate(FEATURES):
    for cls, color in palette.items():
        subset = X_all.loc[y_all == cls, feat]
        axes[i].hist(subset, bins=40, alpha=0.5, label=cls, color=color, density=True)
    axes[i].set_title(feat, fontsize=10)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
axes[0].legend(fontsize=9)
fig.suptitle('Algorithm 3 — Naive Bayes: Class-Conditional Feature Distributions',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo3_nb_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
