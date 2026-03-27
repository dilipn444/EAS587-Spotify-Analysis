# MCP Server

This folder contains the MCP server for the deployed machine learning model.

## Files
- `mcp_server.py` - MCP server implementation
- `../../models/trained_model.pkl` - serialized trained model

## What this server does
The server exposes a `predict` tool that accepts song/audio features and returns a predicted popularity class.

Classes:
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
