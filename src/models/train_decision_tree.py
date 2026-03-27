# ============================================================
# train_decision_tree.py — Algorithm 1: Decision Tree
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Classification (in-class)
# Runtime : < 10 seconds
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
print('=== ALGORITHM 1: Decision Tree ===')
print('Runtime estimate: < 10 seconds\n')
t0 = time.time()

X_train, X_test, y_train, y_test, _ = load_features(scale=False)

dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=SEED)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print('--- Classification Report ---')
print(classification_report(y_test, y_pred, zero_division=0))

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Accuracy   : {acc:.4f}')
print(f'Weighted F1: {f1:.4f}')

# Figure 1A: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=True)
ax.set_title('Algorithm 1 — Decision Tree\nConfusion Matrix (Popularity Tier)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo1_dt_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 1B: Feature Importances
importances = pd.Series(dt.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Algorithm 1 — Decision Tree\nFeature Importances', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo1_dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
