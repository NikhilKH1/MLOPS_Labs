# California Housing Model Card

## Overview

- Run ID: `local_full_20261125`
- Generated (UTC): `2026-11-25T13:29:38.357022+00:00`
- Dataset rows: 20,640
- Train/test rows: 16,512 / 4,128
- Selected model: **hist_gradient_boosting**
- Selection rule: lowest mean cross-validation RMSE
- Data validation: **passed**

The target is median home value for a California census-block group, expressed
in units of $100,000. This educational model is not suitable for appraisal,
lending, insurance, or other high-impact decisions.

## Model selection

Candidates were compared using training-only cross-validation. The held-out test
set was not used to select the winner.

| model | cv_rmse_mean | cv_rmse_std | cv_mae_mean | cv_r2_mean |
|---|---:|---:|---:|---:|
| hist_gradient_boosting | 0.4590 | 0.0134 | 0.3046 | 0.8419 |
| random_forest | 0.5075 | 0.0148 | 0.3323 | 0.8067 |
| elastic_net | 0.7214 | 0.0186 | 0.5287 | 0.6095 |

## Held-out results

| Metric | Value |
|---|---:|
| MAE | 0.3062 |
| RMSE | 0.4642 |
| R-squared | 0.8367 |
| Mean residual | 0.0039 |

## Geographic error slices

These four diagnostic bands are defined by held-out latitude quartiles, so each
contains approximately the same number of rows. They are not demographic
fairness groups.

| region | count | mae | rmse | r2 |
|---|---|---:|---:|---:|
| north_central | 1034 | 0.2820 | 0.4310 | 0.8718 |
| southmost | 1068 | 0.3083 | 0.4507 | 0.8126 |
| south_central | 1003 | 0.3191 | 0.4843 | 0.8375 |
| northmost | 1023 | 0.3159 | 0.4895 | 0.7981 |

## Artifacts

- `training/model.joblib`: complete fitted preprocessing/model pipeline
- `training/model_comparison.csv`: cross-validation results
- `evaluation/metrics.json`: held-out metrics
- `evaluation/predictions.csv`: held-out predictions and residuals
- `evaluation/residual_diagnostics.png`: prediction/residual plots
- `evaluation/permutation_importance.csv` and `.png`: held-out feature importance
- `evaluation/regional_slices.csv`: geographic diagnostic slices

## Limitations

- The data comes from the 1990 U.S. census and does not represent today's market.
- Census-block averages hide variation between individual homes and residents.
- A single deterministic split does not measure every source of uncertainty.
- Geographic features can proxy for socioeconomic and demographic differences.
- Permutation importance describes model reliance, not causal effects.
