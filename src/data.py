"""Data loading and preparation for the bank-churn classification task."""

from pathlib import Path

import pandas as pd

_CANDIDATES = (Path("../data/bank-1.csv"), Path("data/bank-1.csv"))
DATA_PATH = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[-1])

TARGET = "Exited"
# Identifiers with no predictive value.
DROP = ["RowNumber", "CustomerId", "Surname"]


def load_and_clean(path: Path = DATA_PATH):
    """Load the dataset, drop identifiers, return (X, y)."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y
