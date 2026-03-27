# MCP Server

This folder contains the MCP server for the deployed machine learning model.

## Files
- `mcp_server.py` - MCP server implementation
- `../../models/trained_model.pkl` - serialized trained model

## What this server does
The server exposes a `predict` tool that accepts Spotify audio-feature inputs and returns a predicted popularity class.

## Model output classes
- High
- Medium
- Low

## Input features
The prediction tool expects these 8 features:
- danceability
- energy
- valence
- acousticness
- instrumentalness
- tempo
- loudness
- speechiness

## Installation
From the project root:

```bash
pip install -r requirements.txt

```
Dependencies

Main dependencies used for MCP deployment:

mcp[cli]
scikit-learn==1.6.1
numpy
pandas
Run the server

From the project root:
```bash
python src/mcp/mcp_server.py
```

Available tools
predict

Inputs:

- danceability: float
- energy: float
- valence: float
- acousticness: float
- instrumentalness: float
- tempo: float
- loudness: float
- speechiness: float

Returns:

- predicted popularity class
- input features used
- model type
- expected feature list
- class probabilities if supported by the model


Example input:

```bash
{
  "danceability": 0.72,
  "energy": 0.81,
  "valence": 0.65,
  "acousticness": 0.12,
  "instrumentalness": 0.00,
  "tempo": 120.5,
  "loudness": -5.8,
  "speechiness": 0.08
}

```

Example output:

```bash
{
  "prediction": "High",
  "input_features": {
    "danceability": 0.72,
    "energy": 0.81,
    "valence": 0.65,
    "acousticness": 0.12,
    "instrumentalness": 0.0,
    "tempo": 120.5,
    "loudness": -5.8,
    "speechiness": 0.08
  },
  "model_type": "RandomForestClassifier",
  "expected_features": [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
    "loudness",
    "speechiness"
  ]
}
```

model_info

Returns:

- model type
- expected feature list
- class labels
- number of features
- feature names if available
- Validation rules

The server includes basic input validation:

- danceability, energy, valence, acousticness, instrumentalness, and speechiness must be between 0.0 and 1.0
- tempo must be greater than 0
- loudness must be between -80 and 5
- the loaded model must have exactly 8 input features
  
Notes

- The model file must be located at models/trained_model.pkl
- This MCP server is intended for Phase 2 deployment/testing
- The server waits for stdio input after startup, so it may appear to keep running until interrupted.


