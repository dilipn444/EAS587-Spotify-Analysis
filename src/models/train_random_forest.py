# ============================================================
# train_random_forest.py — Algorithm 5: Random Forest
# EAS 587 | Spotify Track Popularity Analysis — Phase 2
# Type    : Classification — ensemble (outside class)
# Source  : Breiman, L. (2001). Random Forests.
#           Machine Learning, 45(1), 5–32.
# Runtime : 3–5 minutes
# ============================================================

import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
print('=== ALGORITHM 5: Random Forest (Outside Algorithm) ===')
print('Source: Breiman (2001). Random Forests. Machine Learning, 45(1).')
print('Runtime estimate: 3–5 minutes\n')
t0 = time.time()

X_train, X_test, y_train, y_test, _ = load_features(scale=False)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print('--- Classification Report ---')
print(classification_report(y_test, y_pred, zero_division=0))

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
print(f'Accuracy   : {acc:.4f}')
print(f'Weighted F1: {f1:.4f}')

# Figure 5A: Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Oranges', colorbar=True)
ax.set_title('Algorithm 5 — Random Forest\nConfusion Matrix (Popularity Tier)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo5_rf_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# Figure 5B: Feature Importances
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='darkorange', edgecolor='black')
plt.title('Algorithm 5 — Random Forest\nFeature Importances', fontsize=13)
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig(f'{BASE}/results/algo5_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model
model_path = f'{BASE}/models/trained_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)
print(f'\n✅ Model saved to: {model_path}')
print(f'   File size: {os.path.getsize(model_path)/1024:.1f} KB')

# Verify reload
with open(model_path, 'rb') as f:
    rf_loaded = pickle.load(f)
assert (rf_loaded.predict(X_test) == y_pred).all(), 'Model reload mismatch!'
print('✅ Model reload verified.')

print(f'\n⏱ Runtime: {time.time()-t0:.1f}s')
