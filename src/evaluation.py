"""Evaluation: the rigor layer for the neural-network classifier."""


def leak_safe_pipeline():
    """Scaling + one-hot encoding + SMOTE inside an imblearn Pipeline.

    TODO (rigor upgrade): SMOTE must be fit on the training fold only. Use an
    imblearn Pipeline (ColumnTransformer -> SMOTE -> KerasClassifier) so resampling
    never sees the validation/test rows.
    """
    raise NotImplementedError


def cross_validate_nn(X, y):
    """Stratified k-fold around the network, not a single split."""
    raise NotImplementedError


def compare_to_trees(X, y):
    """Benchmark the NN against XGBoost/LightGBM. On tabular data, trees usually win."""
    raise NotImplementedError
