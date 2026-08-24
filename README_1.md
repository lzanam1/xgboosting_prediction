# Bankruptcy Prediction Pipeline

Machine learning pipeline that predicts corporate bankruptcy from financial ratios, built as a first-stage credit risk screening filter for banks.

## Overview

- **Dataset:** Polish Companies Bankruptcy Dataset (2000–2013), 5 forecasting horizons (1–5 years)
- **Final model:** XGBoost, tuned with Optuna (50 trials per horizon, F2-score objective), evaluated on the 3-year horizon dataset (predicts bankruptcy 2 years ahead)
- **Missing values:** MICE imputation with a RandomForest estimator (validation R² = 0.871), after winsorizing extreme outliers
- **Interpretability:** SHAP analysis confirms the top drivers are profitability, liquidity, and debt coverage — consistent with how banks assess credit risk

## Key Results (3-year horizon, threshold = 0.1)

| Metric | Value |
|---|---|
| ROC-AUC | 0.862 |
| Recall | 90.9% (90/99 bankruptcies caught) |
| Precision | 10.1% |
| F2 Score | 0.349 |
| Workload reduction | 57.5% (1,209 / 2,101 companies auto-cleared) |

XGBoost outperforms a Random Forest baseline on ROC-AUC, PR-AUC, and recall at a matched decision threshold — see `documentation_en_3_2.pdf` for the full comparison.

## Tech Stack

Python · XGBoost · Optuna · SHAP · scikit-learn · pandas

## How to Run

```bash
git clone https://github.com/lzanam1/bancrucy.git
cd bancrucy
pip install xgboost optuna shap scikit-learn pandas matplotlib scipy
```

Open `bankruptcy_prediction.ipynb` and run the cells top to bottom: data loading → missing value imputation → Optuna hyperparameter tuning → evaluation → SHAP analysis.

## Documentation

- `documentation_en_3_2.pdf` — full methodology and results
- `presentation___final32.pdf` — slide summary of the project

## Limitations

Precision is low at the screening threshold (by design — acceptable for a first-stage filter, not for final lending decisions), features are anonymized (Attr1–Attr64), the data is historical (2000–2013) and may not generalize to current markets, and the train/test split is random rather than time-based.
