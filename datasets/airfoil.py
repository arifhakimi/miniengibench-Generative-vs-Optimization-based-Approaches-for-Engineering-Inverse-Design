from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# UCI Airfoil Self-Noise dataset
# URL format: whitespace-separated; columns documented here for clarity
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
COLUMNS = ["Frequency", "Angle", "Chord", "Velocity", "Thickness", "SPL"]

def load_airfoil(test_size=0.2, val_size=0.1, random_state=42, local_path: str | None = None, return_metadata: bool = False):
    """
    Loads the Airfoil Self-Noise dataset and returns standardized train/val/test splits.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test (all numpy arrays).
        When return_metadata is True, an additional metadata dictionary is appended.

    Notes (Reasoning):
    - Standardized splits ensure fair model comparison.
    - StandardScaler on train only prevents test leakage.
    """
    if local_path and Path(local_path).exists():
        df = pd.read_csv(local_path, sep=r"\s+", header=None, names=COLUMNS)
    else:
        df = pd.read_csv(URL, sep=r"\s+", header=None, names=COLUMNS)

    X = df.drop(columns=["SPL"]).to_numpy(dtype=float)
    y = df["SPL"].to_numpy(dtype=float)

    # Create train/val/test (Reasoning: validation helps tune models without peeking at test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state
    )
    rel = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel, random_state=random_state
    )

    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_test_raw = X_test.copy()

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    if return_metadata:
        metadata = {
            "scaler": scaler,
            "feature_names": COLUMNS[:-1],
            "train_min": X_train_raw.min(axis=0),
            "train_max": X_train_raw.max(axis=0),
        }
        return X_train, y_train, X_val, y_val, X_test, y_test, metadata

    return X_train, y_train, X_val, y_val, X_test, y_test
