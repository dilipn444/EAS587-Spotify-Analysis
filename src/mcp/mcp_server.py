from pathlib import Path
import pickle
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("music-popularity-model")

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "trained_model.pkl"
EXPECTED_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "tempo",
    "loudness",
    "speechiness",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Put trained_model.pkl inside the models/ folder."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if hasattr(model, "n_features_in_") and int(model.n_features_in_) != 8:
    raise ValueError(
        f"Expected model with 8 input features, but found {model.n_features_in_}."
    )


def validate_range(name: str, value: float, min_value: float, max_value: float) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if value < min_value or value > max_value:
        raise ValueError(f"{name} must be between {min_value} and {max_value}.")


@mcp.tool()
def predict(
    danceability: float,
    energy: float,
    valence: float,
    acousticness: float,
    instrumentalness: float,
    tempo: float,
    loudness: float,
    speechiness: float,
) -> dict[str, Any]:
    """
    Predict song popularity class using the trained model.
    Returns: High, Medium, or Low.
    """

    validate_range("danceability", danceability, 0.0, 1.0)
    validate_range("energy", energy, 0.0, 1.0)
    validate_range("valence", valence, 0.0, 1.0)
    validate_range("acousticness", acousticness, 0.0, 1.0)
    validate_range("instrumentalness", instrumentalness, 0.0, 1.0)
    validate_range("speechiness", speechiness, 0.0, 1.0)

    if tempo <= 0:
        raise ValueError("tempo must be greater than 0.")

    if loudness < -80 or loudness > 5:
        raise ValueError("loudness must be in a realistic range (-80 to 5 dB).")

    features = [[
        float(danceability),
        float(energy),
        float(valence),
        float(acousticness),
        float(instrumentalness),
        float(tempo),
        float(loudness),
        float(speechiness),
    ]]

    prediction = model.predict(features)[0]

    result: dict[str, Any] = {
        "prediction": str(prediction),
        "input_features": {
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "tempo": tempo,
            "loudness": loudness,
            "speechiness": speechiness,
        },
        "model_type": type(model).__name__,
        "expected_features": EXPECTED_FEATURES,
    }

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba(features)[0]
        result["class_probabilities"] = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, probabilities)
        }

    return result


@mcp.tool()
def model_info() -> dict[str, Any]:
    """Return basic information about the deployed model."""
    info: dict[str, Any] = {
        "model_type": type(model).__name__,
        "expected_features": EXPECTED_FEATURES,
        "n_features": int(getattr(model, "n_features_in_", 0)),
    }

    if hasattr(model, "feature_names_in_"):
        info["feature_names"] = [str(x) for x in model.feature_names_in_]

    if hasattr(model, "classes_"):
        info["classes"] = [str(x) for x in model.classes_]

    return info


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
