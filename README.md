# 🛡️ AI-Powered Scam Message Detector

A machine learning web app that classifies SMS/WhatsApp-style messages as
**SAFE**, **SUSPICIOUS**, or **SCAM**, built as a university semester project.

## 🎯 Project Overview

This project uses real (non-hardcoded) machine learning to detect scam
messages by analyzing text patterns. It compares three classic ML algorithms
and deploys the best one behind an interactive Streamlit dashboard.

**Pipeline:** Dataset → Google Colab (training) → GitHub (version control) → Streamlit Community Cloud (deployment)

## 🧠 Models Compared

| Model | Why it was chosen |
|---|---|
| Logistic Regression | Fast, interpretable, strong baseline for text classification |
| Multinomial Naive Bayes | Classic, well-proven algorithm for spam/scam text detection |
| Linear SVM | Typically the strongest performer on short-text classification tasks |

All three models are trained on the same TF-IDF features and compared on
accuracy, precision, recall, and F1 score. The best-performing model is
automatically selected and saved for deployment (see `model/metrics.json`
after training for the exact numbers on your data).

## ✨ App Features

- Paste any SMS/WhatsApp message and get an instant classification
- Confidence score and a 0–100 risk meter (LOW / MEDIUM / HIGH)
- Explainable risk indicators (urgency language, suspicious links, requests
  for personal info, etc.)
- Example message buttons to try common scam patterns instantly
- Message statistics (character count, word count, links, punctuation)
- Session-based prediction history

## 📁 Project Structure

```
scam_detector/
├── app.py                          # Streamlit dashboard
├── train_model.py                  # Day 2: initial training script (single split)
├── compare_models.py               # Day 3: cross-validation model comparison
├── validate_best_model.py          # Day 5: KPI + baseline validation of best model
├── MODEL_COMPARISON.md             # Day 3: comparison report (metrics + confusion matrices)
├── VALIDATION_REPORT.md            # Day 5: KPI validation + trade-off decision report
├── requirements.txt                # Python dependencies
├── data/
│   ├── scam_dataset.csv            # Labeled training dataset
│   └── generate_dataset.py         # Script that generated the dataset
├── model/
│   ├── model.pkl                   # Deployed model (CV-selected best: Linear SVM)
│   ├── vectorizer.pkl              # Fitted TF-IDF vectorizer
│   ├── metrics.json                # Final selected model + holdout metrics
│   ├── comparison_results.json     # Full Day 3 cross-validation results (all 3 models)
│   ├── validation_results.json     # Day 5: KPI + baseline validation numbers
│   └── confusion_matrix_*.png      # Confusion matrix per model
└── notebook/
    └── scam_detector_training.ipynb  # Google Colab training notebook
```

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
python train_model.py      # Day 2: initial train/test split comparison
python compare_models.py   # Day 3: cross-validation comparison + finalizes model/model.pkl
streamlit run app.py
```

## 📊 Model Comparison (Day 3)

The three models were re-evaluated using **5-fold stratified cross-validation**
for a more reliable comparison than a single train/test split. Results
(see `MODEL_COMPARISON.md` for the full report):

| Model | Cross-Val F1 (mean) | Cross-Val Accuracy (mean) |
|---|---|---|
| **Linear SVM** ⭐ | 0.986 | 0.987 |
| Logistic Regression | 0.973 | 0.973 |
| Multinomial Naive Bayes | 0.966 | 0.966 |

Linear SVM was the most consistent performer across folds and is the model
currently deployed in `model/model.pkl`.

## ✅ Best Model Validation (Day 5)

The selected model (Linear SVM) was validated against an agreed Sprint 2 KPI
on a fresh hold-out split never used in training or model selection. Full
write-up in `VALIDATION_REPORT.md`.

| KPI | Target | Actual | Result |
|---|---|---|---|
| Minimum accuracy | ≥ 90% | 98.9% | ✅ PASS |
| Minimum weighted F1 | ≥ 90% | 98.9% | ✅ PASS |
| Minimum SCAM recall | ≥ 90% | 100.0% | ✅ PASS |

It was also compared against a majority-class baseline (predicting the most
common label every time), which it beat by **+50.6 points accuracy** and
**+67.4 points weighted F1** — confirming the model learns real language
patterns rather than exploiting class imbalance.

## ☁️ How to Deploy (GitHub → Streamlit Community Cloud)

1. Create a new GitHub repository and push all files in this folder to it
   (including the `model/` folder — the trained `.pkl` files must be committed
   so the deployed app can load them).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, select your repository, branch (`main`), and set the
   main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app.
5. You'll get a public URL like `https://your-app-name.streamlit.app` —
   share this link.

## 🧪 Retraining on Google Colab

If you want to retrain the model (e.g., with a bigger/different dataset):

1. Open `notebook/scam_detector_training.ipynb` in Google Colab.
2. Upload `data/scam_dataset.csv` when prompted.
3. Run all cells — it will train and compare the 3 models automatically.
4. Download the resulting `model.pkl`, `vectorizer.pkl`, and `metrics.json`.
5. Replace the files in your local `model/` folder with the new ones.
6. Commit and push — Streamlit Cloud will redeploy automatically.

## ⚠️ Disclaimer

This is an educational project. Predictions are based on patterns learned
from a limited training dataset and should not be relied on as the sole
method for verifying whether a real message is fraudulent.
