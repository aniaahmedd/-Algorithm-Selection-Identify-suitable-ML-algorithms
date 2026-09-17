# ✅ Testing Checklist

Go through this before submitting/presenting the project.

## Local testing
- [ ] `python train_model.py` runs with no errors and prints metrics for all 3 models
- [ ] `model/model.pkl`, `model/vectorizer.pkl`, `model/metrics.json` are created
- [ ] `streamlit run app.py` launches without errors
- [ ] Each of the 4 example buttons produces a sensible classification
- [ ] Pasting a clearly safe message (e.g. "See you at 6pm") returns SAFE
- [ ] Pasting a clearly scammy message (e.g. a prize/phishing message) returns SCAM
- [ ] Confidence % and risk score (0–100) display correctly
- [ ] Risk indicators section lists relevant flags (or "none detected" for safe messages)
- [ ] Message stats (characters, words, links, exclamation marks) update correctly
- [ ] Analyzing multiple messages builds a session history list
- [ ] "Clear history" button empties the history

## GitHub
- [ ] Repository is public (or shared with your teacher if private)
- [ ] All files pushed: `app.py`, `train_model.py`, `requirements.txt`,
      `README.md`, `.gitignore`, `data/`, `model/`, `notebook/`
- [ ] `model/model.pkl` and `model/vectorizer.pkl` are present in the repo
      (check they're not blocked by `.gitignore`)

## Streamlit Cloud deployment
- [ ] App deploys without build errors
- [ ] Public URL loads and matches local behavior
- [ ] Shared the public URL with your teacher

## Presentation readiness
- [ ] Can explain why 3 algorithms were compared (problem type, data size,
      interpretability, speed — see Day 1 notes)
- [ ] Can explain why the winning model was chosen (metrics in `model/metrics.json`)
- [ ] Can explain what TF-IDF does in plain language
- [ ] Can explain the difference between SAFE / SUSPICIOUS / SCAM in the dataset
- [ ] Can walk through the Streamlit app live during a demo/viva
