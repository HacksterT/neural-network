"""Permutation importance for the churn champion (unresampled HistGBM), measured on test.

HistGradientBoosting exposes no impurity importance, so permutation importance, the drop
in ROC AUC when a feature is shuffled on held-out data, is the honest and only measure here.
"""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .benchmark import SEED, build_preprocessor
from .calibration import split


def feature_importance(seed: int = SEED, n_repeats: int = 20, scoring: str = "roc_auc") -> pd.DataFrame:
    Xtr, Xte, ytr, yte = split(seed)
    pipe = Pipeline(
        [("prep", build_preprocessor()), ("clf", HistGradientBoostingClassifier(random_state=seed))]
    ).fit(Xtr, ytr)
    perm = permutation_importance(
        pipe, Xte, yte, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=-1
    )
    df = pd.DataFrame(
        {"feature": Xte.columns, "perm_importance": perm.importances_mean, "perm_std": perm.importances_std}
    )
    return df.sort_values("perm_importance", ascending=False).reset_index(drop=True).round(4)
