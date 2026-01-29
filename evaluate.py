"""
evaluate.py

Evaluates the trained CAE-based classifier on the test split and prints
accuracy, precision, recall, F1-score and the full classification report.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from model import build_models
from utils.data_loader import load_split
from utils.helpers import load_pickle
from utils.metrics import compute_classification_metrics

CLASSIFIER_WEIGHTS_PATH = "cae_classifier.weights.h5"
ARTIFACTS_PATH = "artifacts.pkl"

def main():
    if not os.path.exists(CLASSIFIER_WEIGHTS_PATH) or not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError("Run train.py first to train and save the classifier.")

    # Load artifacts
    artifacts = load_pickle(ARTIFACTS_PATH)
    feature_cols = artifacts["feature_cols"]
    label_encoder: LabelEncoder = artifacts["label_encoder"]
    scaler = artifacts["scaler"]
    n_features = artifacts["n_features"]
    n_classes = artifacts["n_classes"]

    # Load test data
    df_test = load_split("test")
    X_test = df_test[feature_cols].values
    y_test_str = df_test["label"].values
    y_test = label_encoder.transform(y_test_str)

    # Scale and reshape
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_3d = X_test_scaled[..., np.newaxis]

    # Build models and load classifier weights
    autoencoder, encoder, classifier = build_models(n_features, n_classes)
    classifier.load_weights(CLASSIFIER_WEIGHTS_PATH)

    # Predict
    y_pred_probs = classifier.predict(X_test_scaled_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = compute_classification_metrics(y_test, y_pred, average="weighted")

    print("=== CAE-based Classifier on Test Set ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()
