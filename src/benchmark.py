"""Leak-safe cross-validated benchmark: does the neural network beat trees on tabular data?

Each fold fits the preprocessing and SMOTE on the training portion only, then trains a
neural network, a gradient-boosted tree (sklearn HistGradientBoosting, no libomp needed),
and a logistic baseline, and scores them on the untouched validation fold.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import keras

SEED = 42
NUMERIC = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary", "HasCrCard", "IsActiveMember",
]
CATEGORICAL = ["Geography", "Gender"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def build_nn(input_dim: int) -> keras.Model:
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential(
        [
            keras.layers.Input((input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def _metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, proba),
    }


def cross_val_benchmark(X, y, seed: int = SEED, epochs: int = 40) -> pd.DataFrame:
    """5-fold leak-safe benchmark. Preprocessing + SMOTE fit per training fold."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        pre = build_preprocessor().fit(X.iloc[tr])
        Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
        ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
        Xtr_r, ytr_r = SMOTE(random_state=seed).fit_resample(Xtr, ytr)

        nn = build_nn(Xtr_r.shape[1])
        es = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")
        nn.fit(Xtr_r, ytr_r, validation_split=0.15, epochs=epochs, batch_size=64, verbose=0, callbacks=[es])
        rows.append({"model": "neural_net", "fold": fold, **_metrics(yte, nn.predict(Xte, verbose=0).ravel())})

        hgb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr_r, ytr_r)
        rows.append({"model": "hist_gbm", "fold": fold, **_metrics(yte, hgb.predict_proba(Xte)[:, 1])})

        lr = LogisticRegression(max_iter=1000).fit(Xtr_r, ytr_r)
        rows.append({"model": "logistic", "fold": fold, **_metrics(yte, lr.predict_proba(Xte)[:, 1])})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["roc_auc", "pr_auc", "recall", "precision", "f1", "brier"]
    return df.groupby("model")[cols].agg(["mean", "std"]).round(4)
