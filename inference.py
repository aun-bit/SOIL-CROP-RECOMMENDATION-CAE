"""
inference.py

Loads trained CAE-based classifier and artifacts, then recommends crops for a
single soil sample defined in a JSON file (e.g., sample_input.json).
"""

import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from model import build_models
from utils.helpers import load_pickle

CLASSIFIER_WEIGHTS_PATH = "cae_classifier.weights.h5"
ARTIFACTS_PATH = "artifacts.pkl"

def predict_crop(soil_features: dict, top_k: int = 3):
    # Load artifacts
    artifacts = load_pickle(ARTIFACTS_PATH)
    feature_cols = artifacts["feature_cols"]
    label_encoder: LabelEncoder = artifacts["label_encoder"]
    scaler = artifacts["scaler"]
    n_features = artifacts["n_features"]
    n_classes = artifacts["n_classes"]

    # Build classifier and load weights
    autoencoder, encoder, classifier = build_models(n_features, n_classes)
    classifier.load_weights(CLASSIFIER_WEIGHTS_PATH)

    # Prepare input in correct order
    x = np.array([soil_features[col] for col in feature_cols], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_scaled_3d = x_scaled[..., np.newaxis]

    # Predict probabilities
    probs = classifier.predict(x_scaled_3d)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_labels = label_encoder.inverse_transform(top_indices)
    top_scores = probs[top_indices]

    return list(zip(top_labels, top_scores))

def parse_args():
    parser = argparse.ArgumentParser(description="Soil-Crop Recommendation (CAE-based classifier)")
    parser.add_argument("--input_json", type=str, default="sample_input.json",
                        help="Path to JSON file with soil features")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of crops to recommend")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.input_json, "r") as f:
        soil_features = json.load(f)

    results = predict_crop(soil_features, top_k=args.top_k)
    print("Input soil features:", soil_features)
    print("Top recommendations (crop, probability):")
    for crop, score in results:
        print(f"  {crop}: {score:.4f}")

if __name__ == "__main__":
    main()
