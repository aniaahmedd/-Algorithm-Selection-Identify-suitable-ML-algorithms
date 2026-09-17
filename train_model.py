"""
train_model.py
---------------
AI-Powered Scam Message Detector — Model Training & Comparison

Pipeline:
1. Load labeled dataset (message, label)
2. Split into train/test sets (stratified)
3. Vectorize text with TF-IDF
4. Train three candidate models:
      - Logistic Regression
      - Multinomial Naive Bayes
      - Linear SVM (LinearSVC)
5. Evaluate all three on accuracy, precision, recall, F1
6. Select the best-performing model
7. Save the best model + vectorizer + label encoder + metrics report

This script is designed to be run either locally or inside a
Google Colab notebook (see scam_detector_training.ipynb for the
notebook version with the same steps split into cells).
"""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "data/scam_dataset.csv"
MODEL_DIR = "model"

RANDOM_STATE = 42

def main():
    print("=" * 60)
    print("AI-Powered Scam Message Detector — Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["message", "label"])
    print(f"\nLoaded {len(df)} messages")
    print(df["label"].value_counts())

    X = df["message"].astype(str)
    y = df["label"].astype(str)

    # 2. Train/test split (stratified so class balance is preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 3. TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Define candidate models
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000
        ),
    }

    results = {}
    trained_models = {}

    print("\n" + "-" * 60)
    print("Training & evaluating candidate models")
    print("-" * 60)

    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }
        trained_models[name] = model

        print(f"\n{name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(classification_report(y_test, preds, zero_division=0))

    # 5. Select best model by F1 score
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = trained_models[best_name]

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}  (F1 = {results[best_name]['f1_score']})")
    print("=" * 60)

    # LinearSVC has no predict_proba by default — wrap it with a calibrated
    # classifier so the Streamlit app can always show a confidence percentage,
    # regardless of which model wins.
    if best_name == "Linear SVM":
        print("\nWrapping Linear SVM with probability calibration for confidence scores...")
        calibrated = CalibratedClassifierCV(best_model, cv=3)
        calibrated.fit(X_train_vec, y_train)
        final_model = calibrated
    else:
        final_model = best_model

    # 6. Save artifacts
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "results": results,
                "classes": sorted(y.unique().tolist()),
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            f,
            indent=2,
        )

    print(f"\nSaved model.pkl, vectorizer.pkl, and metrics.json to '{MODEL_DIR}/'")
    print("\nDone! You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    main()
