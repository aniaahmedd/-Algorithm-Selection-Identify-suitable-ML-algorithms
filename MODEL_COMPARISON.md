# Day 3 — Model Comparison Results

Models were compared using **5-fold stratified cross-validation** (more robust than a single train/test split, since every message is used for both training and testing across different folds).

## Primary Metric: Weighted F1 Score

| Rank | Model | F1 (mean ± std) | Accuracy (mean ± std) |
|---|---|---|---|
| 1 | Linear SVM | 0.9863 ± 0.0086 | 0.9865 ± 0.0085 |
| 2 | Logistic Regression | 0.9727 ± 0.0184 | 0.9730 ± 0.0183 |
| 3 | Multinomial Naive Bayes | 0.9657 ± 0.0127 | 0.9662 ± 0.0125 |

## Secondary Metrics (weighted precision / recall, macro F1)

| Model | Precision | Recall | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.9747 | 0.9730 | 0.9697 |
| Multinomial Naive Bayes | 0.9673 | 0.9662 | 0.9602 |
| Linear SVM | 0.9872 | 0.9865 | 0.9807 |

## Selected Model: **Linear SVM**

Selected as the best performer by mean cross-validated weighted F1 score (tie-broken by accuracy).

## Confusion Matrices

Generated from out-of-fold predictions across all cross-validation folds (so every message's prediction comes from a model that never saw it during training). See `model/confusion_matrix_<model_name>.png` for each model.

**Logistic Regression** (rows = actual, columns = predicted)

| actual \\ predicted | SAFE | SCAM | SUSPICIOUS |
|---|---|---|---|
| **SAFE** | 137 | 6 | 0 |
| **SCAM** | 0 | 217 | 0 |
| **SUSPICIOUS** | 4 | 2 | 78 |

**Multinomial Naive Bayes** (rows = actual, columns = predicted)

| actual \\ predicted | SAFE | SCAM | SUSPICIOUS |
|---|---|---|---|
| **SAFE** | 137 | 6 | 0 |
| **SCAM** | 0 | 217 | 0 |
| **SUSPICIOUS** | 6 | 3 | 75 |

**Linear SVM** (rows = actual, columns = predicted)

| actual \\ predicted | SAFE | SCAM | SUSPICIOUS |
|---|---|---|---|
| **SAFE** | 143 | 0 | 0 |
| **SCAM** | 0 | 217 | 0 |
| **SUSPICIOUS** | 6 | 0 | 78 |

## Per-Class Report — Selected Model

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SAFE | 0.960 | 1.000 | 0.979 | 143 |
| SCAM | 1.000 | 1.000 | 1.000 | 217 |
| SUSPICIOUS | 1.000 | 0.929 | 0.963 | 84 |

## Notes / Residual Review

- The SUSPICIOUS class consistently shows the most confusion with the other two classes across all models — expected, since suspicious messages sit linguistically between clearly safe and clearly fraudulent text.
- Misclassifications rarely occur between SAFE and SCAM directly (the two most distinct classes); most errors involve SUSPICIOUS being predicted as SAFE or SCAM, or vice versa.
- Cross-validation standard deviations are small across folds, indicating the results are stable and not driven by a lucky/unlucky single split.
