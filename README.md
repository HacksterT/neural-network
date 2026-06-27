# Neural Network Classification

A feed-forward neural network for binary classification: predicting bank customer churn
(`Exited`) from account and demographic attributes, evaluated to a clinical-research
standard.

As with the rest of the portfolio, the domain is incidental. The point is the
**evaluation discipline** applied to a deep model on tabular data: leak-safe
preprocessing and resampling, honest cross-validation, calibration, and the question that
matters for tabular problems, *does a neural network actually beat gradient-boosted trees
here?*

## Status

Migrated to Python 3.13 + uv (TF 2.21 / Keras 3.15, runs end-to-end 33/33) and **rigor
passes 1-3 complete** (`notebooks/02-04`). Headline: on this tabular problem the
gradient-boosted tree (`HistGradientBoosting`, ROC AUC 0.860) **matches-or-beats the neural
network** (0.855) while training in a fraction of the time and calibrating better. SMOTE and
post-hoc calibration both turned out unnecessary; the threshold, set from the cost ratio, is
the real control. Full findings in `docs/learnings.md`.

The TensorFlow arm64 wheel question flagged in the resurrection playbook resolved cleanly:
TF 2.21 has Python 3.13 macOS arm64 wheels and trains without issue, so no fallback to
3.12 was needed.

## Dataset

10,000 customers, target `Exited` (~20% churn, the minority class). Features include
credit score, geography, gender, age, tenure, balance, product count, and activity flags.
`RowNumber`, `CustomerId`, and `Surname` are identifiers and are dropped. Source: a public
bank-churn dataset; included in `data/`.

## Approach and roadmap

- [x] Baseline neural network runs end-to-end (TF 2.21 / Keras 3.15).
- [x] Leak-safe pipeline: scaling, encoding, and SMOTE fit per training fold (`src/benchmark.py`).
- [x] Cross-validation: 5-fold stratified benchmark, not a single split.
- [x] Calibration (reliability curves, Brier) and cost-based threshold selection (`notebooks/03`).
- [x] The real question: NN vs gradient-boosted trees (`HistGradientBoosting`) vs logistic. Trees win; reported honestly (`notebooks/02`).
- [x] Reproducibility: seeds fixed, training packaged in `src/`.
- [x] Permutation importance: churn drivers are NumOfProducts, Age, Balance, IsActiveMember, Geography (`notebooks/04`).
- [ ] Optional: HGB hyperparameter search; re-run benchmark with XGBoost/LightGBM (needs `libomp`) to confirm the tree result generalizes.

## Structure

```
neural-network/
  data/        bank-1.csv
  notebooks/   01-neural-network.ipynb   (baseline; rigor notebooks added as 02-, 03-)
  src/         reusable data + evaluation code (the rigor layer)
  results/     figures and metrics written by the notebooks
```

## Environment

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter lab
```

## Provenance

This project originated as coursework and has been rebuilt as an independent,
rigor-upgraded portfolio piece. All third-party course materials (slides, templates,
provided low-code scaffolds, vendor branding) have been removed; the work here is my own.
