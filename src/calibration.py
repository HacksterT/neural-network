"""Calibration and operating-point analysis for the churn champion (HistGBM).

SMOTE improves recall but inflates predicted probabilities (the model is trained as if
churn were 50%), wrecking calibration. This module compares the SMOTE model against an
unresampled model with and without post-hoc calibration on a held-out test split, and
chooses the decision threshold from the cost of errors instead of the default 0.5.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .benchmark import SEED, build_preprocessor
from .data import load_and_clean


def split(seed: int = SEED):
    X, y = load_and_clean()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def fit_probabilities(seed: int = SEED):
    """Return (y_test, {name: proba}) for SMOTE / unweighted / calibrated HGB."""
    Xtr, Xte, ytr, yte = split(seed)
    pre = build_preprocessor().fit(Xtr)
    Atr, Ate = pre.transform(Xtr), pre.transform(Xte)
    ytr = ytr.to_numpy()
    yte = yte.reset_index(drop=True)

    out = {}
    Ar, yr = SMOTE(random_state=seed).fit_resample(Atr, ytr)
    out["smote"] = HistGradientBoostingClassifier(random_state=seed).fit(Ar, yr).predict_proba(Ate)[:, 1]
    out["no_smote"] = HistGradientBoostingClassifier(random_state=seed).fit(Atr, ytr).predict_proba(Ate)[:, 1]
    out["no_smote_isotonic"] = (
        CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=seed), method="isotonic", cv=5)
        .fit(Atr, ytr)
        .predict_proba(Ate)[:, 1]
    )
    return yte, out


def brier_table(yte, probs: dict) -> pd.DataFrame:
    rows = [
        {"model": k, "brier": brier_score_loss(yte, p), "roc_auc": roc_auc_score(yte, p)}
        for k, p in probs.items()
    ]
    return pd.DataFrame(rows).round(4)


def reliability(yte, probs: dict, n_bins: int = 10) -> dict:
    out = {}
    for k, p in probs.items():
        frac, mean = calibration_curve(yte, p, n_bins=n_bins, strategy="quantile")
        out[k] = (mean, frac)
    return out


def threshold_sweep(yte, proba) -> pd.DataFrame:
    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(yte, pred, zero_division=0),
                "recall": recall_score(yte, pred, zero_division=0),
                "f1": f1_score(yte, pred, zero_division=0),
                "flagged": int(pred.sum()),
            }
        )
    return pd.DataFrame(rows).round(3)


def threshold_by_cost(yte, proba, fn_cost: float = 5, fp_cost: float = 1) -> dict:
    """Churn is asymmetric: losing a customer (FN) usually costs more than a retention offer (FP)."""
    best = None
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        cost = fn * fn_cost + fp * fp_cost
        if best is None or cost < best["cost"]:
            best = {
                "threshold": round(float(t), 3),
                "cost": float(cost),
                "fn": int(fn),
                "fp": int(fp),
                "recall": round(recall_score(yte, pred, zero_division=0), 3),
                "precision": round(precision_score(yte, pred, zero_division=0), 3),
            }
    return best
