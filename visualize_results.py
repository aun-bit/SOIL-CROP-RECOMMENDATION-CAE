"""
visualize_results.py

Generates and saves AE loss curves, classifier loss/accuracy curves,
confusion matrix, and micro-averaged ROC/PR curves into docs/figures/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize

from model import build_models
from utils.helpers import load_pickle
from utils.data_loader import load_split

CLASSIFIER_WEIGHTS_PATH = "cae_classifier.weights.h5"
ARTIFACTS_PATH = "artifacts.pkl"
HISTORY_PATH = "training_history.pkl"
FIG_DIR = os.path.join("docs", "figures")


def plot_ae_loss(history):
    ae_loss = history.get("ae_loss", [])
    ae_val_loss = history.get("ae_val_loss", [])

    if not ae_loss:
        print("No autoencoder loss history found.")
        return

    epochs = range(1, len(ae_loss) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, ae_loss, label="Train Loss")
    plt.plot(epochs, ae_val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(FIG_DIR, "ae_loss_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved AE loss curve to {out_path}")


def plot_classifier_curves(history):
    cls_loss = history.get("cls_loss", [])
    cls_val_loss = history.get("cls_val_loss", [])
    cls_acc = history.get("cls_accuracy", [])
    cls_val_acc = history.get("cls_val_accuracy", [])

    if not cls_loss:
        print("No classifier loss/accuracy history found.")
        return

    epochs = range(1, len(cls_loss) + 1)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, cls_loss, label="Train Loss")
    plt.plot(epochs, cls_val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classifier Loss")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(FIG_DIR, "classifier_loss_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved classifier loss curve to {out_path}")

    # Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, cls_acc, label="Train Accuracy")
    plt.plot(epochs, cls_val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(FIG_DIR, "classifier_accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved classifier accuracy curve to {out_path}")


def plot_confusion_matrix_img(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_roc_pr_curves(y_true, y_score, n_classes):
    """
    y_true: integer labels, shape (n_samples,)
    y_score: predicted probabilities, shape (n_samples, n_classes)
    """
    # Binarize labels for multi-class micro-averaged ROC/PR
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    # Micro-averaged ROC
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="blue", lw=2,
             label=f"micro-average ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    out_path = os.path.join(FIG_DIR, "roc_curve_micro.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve (micro-average) to {out_path}")

    # Micro-averaged Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    ap = average_precision_score(y_true_bin, y_score, average="micro")

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color="green", lw=2,
             label=f"micro-average PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    out_path = os.path.join(FIG_DIR, "precision_recall_curve_micro.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Precision-Recall curve (micro-average) to {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load training history
    history = load_pickle(HISTORY_PATH)

    # Plot training curves
    plot_ae_loss(history)
    plot_classifier_curves(history)

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

    # Build classifier and load weights
    _, _, classifier = build_models(n_features, n_classes)
    classifier.load_weights(CLASSIFIER_WEIGHTS_PATH)

    # Predict probabilities and labels
    y_pred_probs = classifier.predict(X_test_scaled_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Plot confusion matrix
    class_names = label_encoder.classes_
    plot_confusion_matrix_img(y_test, y_pred, class_names)

    # Plot ROC and PR curves (micro-average)
    plot_roc_pr_curves(y_test, y_pred_probs, n_classes)


if __name__ == "__main__":
    main()
