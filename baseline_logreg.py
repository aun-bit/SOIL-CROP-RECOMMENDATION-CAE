# baseline_logreg.py
"""
Baseline model: LogisticRegression on raw (scaled) soil features

Uses the same train/val/test splits as the CAE model.
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils.data_loader import FEATURE_COLS, TARGET_COL, load_split, create_splits
from utils.metrics import compute_classification_metrics

def main():
    # Ensure train/val/test splits exist
    data_dir = os.path.join("data", "soil_crop")
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        print("Splits not found. Creating train/val/test splits...")
        create_splits()

    # Load splits
    df_train = load_split("train")
    df_val   = load_split("val")
    df_test  = load_split("test")

    # Combine train + val for stronger baseline training
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    X_train = df_trainval[FEATURE_COLS].values
    y_train_str = df_trainval[TARGET_COL].values

    X_test = df_test[FEATURE_COLS].values
    y_test_str = df_test[TARGET_COL].values

    # Label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_test  = label_encoder.transform(y_test_str)

    # Scale features (logistic regression benefits from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Logistic Regression baseline (simple, compatible with most versions)
    clf = LogisticRegression(
        max_iter=500,
        solver="lbfgs"   # works for multinomial by default when classes > 2
    )

    print("Training LogisticRegression baseline on raw (scaled) features...")
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    metrics = compute_classification_metrics(y_test, y_pred, average="weighted")

    print("=== Baseline: LogisticRegression on raw features ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])

if __name__ == "__main__":
    main()