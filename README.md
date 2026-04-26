# EAS 587 – Spotify Track Popularity Analysis

**Course:** EAS 587 – Introduction to Data Science (Spring 2026)  
**Team:** TriForce  
**Phase:** 3 — Scalable Data Processing & ML Pipeline

---

## Team Members

| Name | UB Person Number |
|---|---|
| Dilip Nallamasa | 50668608 |
| Harsha Adinarayanaraju Lolabattu | 50682313 |
| Pamulapati Venkat Sai Pavan | 50660304 |

---

## Project Overview

This project applies machine learning and statistical modeling to a Spotify tracks dataset to predict and understand song **popularity tiers** (Low / Medium / High) based on audio features such as danceability, energy, tempo, valence, and more.

Phase 2 builds directly on Phase 1's cleaned and preprocessed dataset (89,726 tracks after removing duplicates and nulls from ~114,000 raw rows). Six algorithms are applied, one model is deployed as an MCP server, and all results are connected back to the Phase 1 problem statement.

---

## Repository Structure

```
EAS587-Spotify-Analysis/
├── README.md                       ← This file (updated for Phase 2)
├── requirements.txt                ← All dependencies (Phase 1 + Phase 2)
├── .gitignore
│
├── data/
│   ├── raw/                        ← Original Kaggle dataset
│   └── processed/                  ← Cleaned dataset from Phase 1
│
├── models/
│   └── trained_model.pkl           ← Serialized Random Forest model (Phase 2)
│
├── reports/
│   └── figures/                    ← All generated visualizations
│
└── src/
    ├── data_collection.py
    ├── data_cleaning.py
    ├── eda.py
    │
    ├── models/
    │   ├── train_decision_tree.py
    │   ├── train_knn.py
    │   ├── train_naive_bayes.py
    │   ├── train_kmeans.py
    │   ├── train_random_forest.py
    │   └── train_pca.py
    │
    ├── notebooks/
    │   └── databricks/
    │       ├── 01_bronze_layer.ipynb
    │       ├── 02_silver_layer.ipynb
    │       ├── 03_gold_layer.ipynb
    │       ├── 04_mllib_models.ipynb
    │       └── 05_additional_source.ipynb
    │
    └── mcp/
        ├── mcp_server.py
        └── README.md
```

---

## Phase 1 Summary

- **Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) via Kaggle
- **Raw rows:** ~114,000 tracks
- **Usable rows after cleaning:** 89,726
- **Target variable engineered:** `popularity_tier` — Low / Medium / High, derived from Spotify's raw 0–100 popularity score
- **Audio features used:** `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `tempo`, `loudness`, `speechiness`
- **Phase 1 use cases addressed:**
  1. Predict whether a track will be popular before release
  2. Identify which audio features drive popularity
  3. Cluster tracks into natural listening segments

---

## Phase 2 — Algorithms Applied

Six algorithms were applied, four from class and two from outside:

| # | Algorithm | Type | Source |
|---|---|---|---|
| 1 | Decision Tree | Classification (in-class) | Scikit-learn |
| 2 | k-Nearest Neighbors (k-NN) | Classification (in-class) | Scikit-learn |
| 3 | Naive Bayes | Classification (in-class) | Scikit-learn |
| 4 | k-Means Clustering | Clustering (in-class) | Scikit-learn |
| 5 | Random Forest | Ensemble Classification (outside) | Breiman (2001) |
| 6 | PCA | Dimensionality Reduction (outside) | Jolliffe (2002) |

All scripts are in `src/models/`. Each script is self-contained, sets `random_state=42`, and saves its outputs to `reports/figures/`.

---

## Phase 2 — Setup & Installation

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/dilipn444/EAS587-Spotify-Analysis.git
cd EAS587-Spotify-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The dataset (89,726 rows) and some ML algorithms (especially k-NN) require meaningful compute time. See runtime estimates below. Plan accordingly when running in full.

---

## Phase 2 — Running the Models

Each algorithm can be run independently from the project root. All random seeds are fixed to `42` for reproducibility.

```bash
python src/models/train_decision_tree.py
python src/models/train_knn.py
python src/models/train_naive_bayes.py
python src/models/train_kmeans.py
python src/models/train_random_forest.py
python src/models/train_pca.py
```

### Expected Runtimes

| Script | Estimated Runtime |
|---|---|
| `train_decision_tree.py` | < 10 seconds |
| `train_naive_bayes.py` | ~20 seconds |
| `train_kmeans.py` | ~13 seconds |
| `train_random_forest.py` | ~14 seconds |
| `train_pca.py` | ~4 seconds |
| `train_knn.py` | ~143 seconds |

All scripts print progress indicators for long-running operations.

### Outputs

Each script saves:
- A visualization (confusion matrix, cluster plot, PCA projection, etc.) to `reports/figures/`
- A summary of performance metrics to stdout

The Random Forest script additionally serializes the trained model to `models/trained_model.pkl` for use by the MCP server.

---

## MCP Deployment

One trained model — **Random Forest** — is deployed as an MCP (Model Context Protocol) server, making it callable by AI assistants such as Claude Desktop.

> Full MCP setup instructions are also available in [`src/mcp/README.md`](src/mcp/README.md).

### Prerequisites

Ensure the trained model file exists before starting the server:

```bash
python src/models/train_random_forest.py
# Confirms: models/trained_model.pkl created
```

### Start the MCP Server

```bash
python src/mcp/mcp_server.py
```

The server starts and listens for stdio input. It exposes two tools:

---

### Available MCP Tools

#### `predict`

Accepts 8 audio feature values and returns a predicted popularity tier.

**Input parameters:**

| Parameter | Type | Range | Description |
|---|---|---|---|
| `danceability` | float | 0.0 – 1.0 | How suitable a track is for dancing |
| `energy` | float | 0.0 – 1.0 | Intensity and activity level |
| `valence` | float | 0.0 – 1.0 | Musical positiveness |
| `acousticness` | float | 0.0 – 1.0 | Confidence the track is acoustic |
| `instrumentalness` | float | 0.0 – 1.0 | Predicts whether a track has no vocals |
| `tempo` | float | > 0 | Estimated tempo in BPM |
| `loudness` | float | -80 to 5 | Overall loudness in dB |
| `speechiness` | float | 0.0 – 1.0 | Presence of spoken words |

**Returns:**

```json
{
  "prediction": "High",
  "class_probabilities": {"Low": 0.08, "Medium": 0.21, "High": 0.71},
  "input_features": { ... },
  "model_type": "RandomForestClassifier"
}
```

#### `model_info`

Returns metadata about the deployed model.

```json
{
  "model_type": "RandomForestClassifier",
  "features": ["danceability", "energy", "valence", "acousticness",
               "instrumentalness", "tempo", "loudness", "speechiness"],
  "output_classes": ["Low", "Medium", "High"],
  "n_features": 8
}
```

---

### Using with Claude Desktop

1. Download Claude Desktop: https://claude.ai/download
2. Configure your `claude_desktop_config.json` to point to `src/mcp/mcp_server.py`
3. Start the server and connect Claude Desktop
4. Example prompt to Claude:

> *"Predict the popularity of a track with danceability 0.8, energy 0.9, valence 0.6, acousticness 0.1, instrumentalness 0.0, tempo 128, loudness -5, speechiness 0.05"*

Claude will call the `predict` tool and return the tier with class probabilities.

---

## Reproducibility

- **All random seeds** are set to `42` across every script that uses randomization (train/test splits, k-Means initialization, Random Forest, Decision Tree)
- **Pipeline verified end-to-end** in a fresh Google Colab session by a second team member — all outputs matched reference results within floating-point tolerance
- **MCP server verified** by cloning the repository fresh, installing from `requirements.txt`, running `train_random_forest.py` to generate the `.pkl`, and confirming the server started and responded correctly

> ✅ Dry-run verification completed. See report for confirmation statement.

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML algorithms (Decision Tree, k-NN, Naive Bayes, k-Means, Random Forest, PCA) |
| `matplotlib`, `seaborn` | Visualizations |
| `fastmcp` | MCP server framework |
| `pickle` (stdlib) | Model serialization |
| `joblib` | Efficient model persistence |

Install everything with:

```bash
pip install -r requirements.txt
```
---

## Phase 3 — Databricks, Spark MLlib & Delta Lake

Phase 3 scales the Spotify popularity prediction pipeline using **Databricks**, **Apache Spark**, **Spark MLlib**, and **Delta Lake**, implementing a full **Medallion Architecture (Bronze → Silver → Gold)**.

---

## Databricks Notebooks

All Phase 3 notebooks are located at:

src/notebooks/databricks/


Execution order:

1. `01_bronze_layer.ipynb`
2. `02_silver_layer.ipynb`
3. `03_gold_layer.ipynb`
4. `04_mllib_models.ipynb`
5. `05_additional_source.ipynb`

---

## Data Storage (Unity Catalog Volume)

Raw dataset stored in:
/Volumes/workspace/default/spotify_data/raw_datset.csv


All processed data is stored as **Delta Tables**.

---

## Medallion Architecture

### Bronze Layer

- Table: `bronze_spotify`
- Raw ingestion from CSV
- No transformations applied

---

### Silver Layer

- Table: `silver_spotify`
- Data cleaning and preprocessing:
  - Removed duplicates
  - Dropped index column
  - Fixed malformed numeric values using `try_cast`
  - Removed invalid/null feature rows

---

### Gold Layer

Tables created:

- `gold_genre_metrics`
- `gold_spotify_ml_features`

Features:
- Aggregated genre-level insights
- ML-ready dataset
- Created target variable `popularity_tier`

---

## Spark MLlib Models

Two models were implemented using Spark MLlib:

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Decision Tree | 0.5917 | 0.5555 |
| Random Forest | 0.6222 | 0.5777 |

Comparison table saved as:
gold_model_comparison


Models saved to:
/Volumes/workspace/default/spotify_data/dt_model
/Volumes/workspace/default/spotify_data/rf_model


---

## Additional Data Source (Task 2)

An artist-level dataset was derived and joined with track-level data.

Table created:
gold_combined_dataset


### Insights:

1. Artist average popularity improves prediction context  
2. Frequent artists show stable popularity trends  
3. Aggregated features enhance model input quality  

---

## Technologies Used (Phase 3)

- Databricks
- Apache Spark
- PySpark
- Spark MLlib
- Delta Lake


---

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly Media.
5. Dataset: https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset
6. MCP Protocol Documentation: https://modelcontextprotocol.io/
7. Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
8. Scikit-learn Documentation: https://scikit-learn.org/stable/

---

*EAS 587 — Spring 2026 | University at Buffalo*
