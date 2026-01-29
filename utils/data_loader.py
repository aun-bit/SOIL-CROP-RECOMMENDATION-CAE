"""
data_loader.py

Handles loading Crop_recommendation.csv, creating stratified train/val/test
splits, and providing helper functions to load each split.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .helpers import ensure_dir

DATA_DIR = os.path.join("data", "soil_crop")
RAW_CSV = os.path.join(DATA_DIR, "Crop_recommendation.csv")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"

def load_raw_dataset(csv_path: str = RAW_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            f"Download Crop_recommendation.csv from Kaggle "
            f"and place it there."
        )
    df = pd.read_csv(csv_path)
    return df

def create_splits(test_size=0.15, val_size=0.15, random_state=42):
    """
    Create train/val/test splits and save them as CSV in data/soil_crop/.
    """
    ensure_dir(DATA_DIR)
    df = load_raw_dataset(RAW_CSV)

    # First split off test
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET_COL]
    )

    # Then split train/val
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        stratify=df_trainval[TARGET_COL]
    )

    df_train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    df_test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print("Splits created:")
    print("  train:", df_train.shape)
    print("  val:  ", df_val.shape)
    print("  test: ", df_test.shape)

def load_split(split: str) -> pd.DataFrame:
    """
    split: 'train' | 'val' | 'test'
    """
    path = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run create_splits() first (python -m utils.data_loader)."
        )
    return pd.read_csv(path)

if __name__ == "__main__":
    # Allow: python -m utils.data_loader to create splits
    create_splits()
