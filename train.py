import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

from model import build_models
from utils.data_loader import FEATURE_COLS, TARGET_COL, load_split, create_splits
from utils.helpers import save_pickle

np.random.seed(42)
tf.random.set_seed(42)

AE_WEIGHTS_PATH = "autoencoder.weights.h5"
CLASSIFIER_WEIGHTS_PATH = "cae_classifier.weights.h5"
HISTORY_PATH = "training_history.pkl"
ARTIFACTS_PATH = "artifacts.pkl"

def main():
    # Ensure train/val/test splits exist
    if not os.path.exists(os.path.join("data", "soil_crop", "train.csv")):
        print("Creating train/val/test splits...")
        create_splits()

    # Load train and val
    df_train = load_split("train")
    df_val   = load_split("val")

    X_train = df_train[FEATURE_COLS].values
    y_train_str = df_train[TARGET_COL].values

    X_val   = df_val[FEATURE_COLS].values
    y_val_str = df_val[TARGET_COL].values

    # Label encoding (supervised labels for classifier)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    y_val   = label_encoder.transform(y_val_str)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    n_features = X_train_scaled.shape[1]
    n_classes  = len(label_encoder.classes_)

    # Reshape for Conv1D: (samples, timesteps, channels)
    X_train_scaled_3d = X_train_scaled[..., np.newaxis]
    X_val_scaled_3d   = X_val_scaled[..., np.newaxis]

    # Build models
    autoencoder, encoder, classifier = build_models(n_features, n_classes)

    # ----- Stage 1: Autoencoder pretraining -----
    autoencoder.compile(optimizer="adam", loss="mse")

    print("Pretraining autoencoder...")
    history_ae = autoencoder.fit(
        X_train_scaled_3d, X_train_scaled_3d,
        validation_data=(X_val_scaled_3d, X_val_scaled_3d),
        epochs=20,
        batch_size=32,
        shuffle=True,
        verbose=1
    )
    autoencoder.save_weights(AE_WEIGHTS_PATH)
    print(f"Saved autoencoder weights to {AE_WEIGHTS_PATH}")

    # ----- Stage 2: Classifier training (CAE-based model) -----
    classifier.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Training classifier on top of CAE encoder...")
    history_cls = classifier.fit(
        X_train_scaled_3d, y_train,
        validation_data=(X_val_scaled_3d, y_val),
        epochs=50,
        batch_size=32,
        shuffle=True,
        verbose=1
    )
    classifier.save_weights(CLASSIFIER_WEIGHTS_PATH)
    print(f"Saved classifier weights to {CLASSIFIER_WEIGHTS_PATH}")

    # Save training history
    full_history = {
        "ae_loss": history_ae.history.get("loss", []),
        "ae_val_loss": history_ae.history.get("val_loss", []),
        "cls_loss": history_cls.history.get("loss", []),
        "cls_val_loss": history_cls.history.get("val_loss", []),
        "cls_accuracy": history_cls.history.get("accuracy", []),
        "cls_val_accuracy": history_cls.history.get("val_accuracy", []),
    }
    save_pickle(full_history, HISTORY_PATH)
    print(f"Saved training history to {HISTORY_PATH}")

    # Save artifacts for evaluation & inference
    artifacts = {
        "feature_cols": FEATURE_COLS,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "n_features": n_features,
        "n_classes": n_classes,
    }
    save_pickle(artifacts, ARTIFACTS_PATH)
    print(f"Saved artifacts to {ARTIFACTS_PATH}")

if __name__ == "__main__":
    main()
