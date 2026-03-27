# EAS 587 – Phase 2  
# Spotify Track Popularity Analysis

## Team Name: TriForce

### Team Members
1. Dilip Nallamasa (UB Person Number: 50668608)
2. Harsha Adinarayanaraju Lolabattu (UB Person Number: 50682313)
3. Pamulapati Venkat Sai Pavan (UB Person Number: 50660304)

---

## Project Overview

This project analyzes Spotify track audio features to understand and predict song
popularity tiers (Low / Medium / High) using machine learning and statistical modeling.

Phase 2 builds on Phase 1's cleaned dataset and applies 6 algorithms:

1. Decision Tree (in-class)
2. k-Nearest Neighbors (in-class)
3. Naive Bayes (in-class)
4. k-Means Clustering (in-class)
5. Random Forest (outside — Breiman 2001)
6. PCA — Principal Component Analysis (outside — Jolliffe 2002)

One trained model (Random Forest) is deployed as an MCP server.

## Phase 1 Summary

- Data acquired from Kaggle (Spotify Tracks Dataset)
- Cleaned and preprocessed 114,000 raw tracks → 89,726 usable rows
- Engineered popularity_tier column (Low / Medium / High)
- Performed full EDA on 8 audio features

---

## Phase 2 — How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run individual algorithm scripts

Each script can also be run standalone from the project root:
```bash
python src/models/train_decision_tree.py
python src/models/train_knn.py
python src/models/train_naive_bayes.py
python src/models/train_kmeans.py
python src/models/train_random_forest.py
python src/models/train_pca.py
```

All random seeds are set to 42. Expected runtimes:
- Decision Tree: < 10 seconds
- k-NN: ~143 seconds
- Naive Bayes: ~20 seconds
- k-Means: ~13 seconds
- Random Forest: ~14 seconds
- PCA: ~4 seconds

---

## MCP Deployment — How to Run

The trained Random Forest model is served as an MCP tool via FastMCP.

### Setup
```bash
pip install -r requirements.txt
```

Make sure models/trained_model.pkl exists (run train_random_forest.py first).

### Start the server
```bash
python src/mcp/mcp_server.py
```

### Available tools

**predict** — accepts 8 audio feature values, returns predicted popularity tier

danceability     float   0.0 – 1.0
energy           float   0.0 – 1.0
valence          float   0.0 – 1.0
acousticness     float   0.0 – 1.0
instrumentalness float   0.0 – 1.0
tempo            float   > 0
loudness         float   -80 to 5 dB
speechiness      float   0.0 – 1.0

Returns: prediction (High / Medium / Low), class_probabilities, input_features, model_type

**model_info** — returns model type, feature names, output classes, number of features

### Example call (Claude Desktop)

Once the server is running and connected to Claude Desktop, you can ask:

"Predict the popularity of a track with danceability 0.8, energy 0.9, valence 0.6,
acousticness 0.1, instrumentalness 0.0, tempo 128, loudness -5, speechiness 0.05"

---

## Reproducibility

- All random seeds set to 42 across all scripts
- Pipeline verified end-to-end in a fresh Google Colab session by a second team member
- All outputs matched reference results within floating-point tolerance
- The MCP server was tested by cloning the repository, installing dependencies from requirements.txt, and running python src/mcp/mcp_server.py
- The server started successfully and waited for stdio input as expected

---

## References

- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
- Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12.
- Dataset: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset
- MCP Protocol: https://modelcontextprotocol.io/
