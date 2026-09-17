# Day 5 — Best Model Validation Report

This report validates the selected model against the project's agreed KPI and against a naive baseline, using a fresh hold-out validation split that played no role in model training or model selection (Day 2 and Day 3 both used a different random seed).

## 1. Agreed KPI (Sprint 2)

| KPI | Target | Actual | Result |
|---|---|---|---|
| Minimum accuracy | ≥ 90% | 98.9% | ✅ PASS |
| Minimum weighted F1 | ≥ 90% | 98.9% | ✅ PASS |
| Minimum SCAM recall (catch real scams) | ≥ 90% | 100.0% | ✅ PASS |

**Overall KPI result: ✅ PASSED**

SCAM recall was set as its own KPI (not just overall accuracy) because for this use case, missing an actual scam (a false negative) is more costly than a false alarm on a suspicious message — the app should err on the side of catching real fraud. Note: a naive always-predict-SCAM baseline can trivially satisfy this one KPI in isolation, which is why it must always be read together with weighted F1 and accuracy, not alone.

## 2. Best Model vs. Baseline

**Baseline:** a majority-class predictor (always predicts the single most common label in the training data). This represents the "do-nothing-smart" floor — any real model must clearly beat it.

| Metric | Baseline | Linear SVM | Improvement |
|---|---|---|---|
| Accuracy | 48.3% | 98.9% | +50.6 pts |
| Weighted F1 | 31.5% | 98.9% | +67.4 pts |
| SCAM recall | 100.0% | 100.0% | +0.0 pts |

The baseline reaches 48.3% accuracy and a trivial 100% SCAM recall only because SCAM happens to be the majority class in this split — it predicts SCAM for every single message, so it "catches" all real scams but also wrongly flags every SAFE and SUSPICIOUS message as a scam (hence its very low weighted F1). Linear SVM matches that recall on real scams while actually distinguishing between the three classes, which is what its much higher accuracy and F1 demonstrate.

## 3. Model Trade-offs (Recap from Day 3)

| Model | Strengths | Weaknesses | Verdict |
|---|---|---|---|
| **Linear SVM** | Highest & most stable CV F1 (98.6%); best margin separation for short, templated text | No native probability output (needs calibration wrapper for confidence %); slightly less interpretable than Logistic Regression | ✅ **Selected** — best raw performance, calibration overhead is a one-time, solved cost |
| Logistic Regression | Fast, highly interpretable (coefficients show which words drive each class); native probabilities | ~1.3 pts lower F1 than SVM under cross-validation | Strong runner-up; would be the pick if interpretability outweighed raw accuracy |
| Multinomial Naive Bayes | Fastest to train, simplest, classic spam-filter baseline | Lowest F1 of the three; weakest at separating SUSPICIOUS from the other classes | Useful sanity-check baseline, not selected for deployment |

## 4. Decision

**Linear SVM is confirmed as the production model.** It passes every agreed KPI on a fresh, previously unseen validation split, and its cross-validated performance (Day 3) was both the highest and the most consistent across folds. The lack of native probability output was mitigated with `CalibratedClassifierCV`, so the deployed app can still show a confidence percentage to users.

## 5. Residual Risk / Known Limitations

- SUSPICIOUS remains the hardest class to separate from SAFE/SCAM for every model tested — flagged as a limitation, not a blocker, since SCAM recall (the priority KPI) is unaffected.
- The dataset is template-generated rather than sourced from real-world messages, so real-world performance should be re-validated once actual user-reported messages are available.
- KPI targets were set by the project team for Sprint 2 and can be revisited once a larger, real-world dataset is collected.
