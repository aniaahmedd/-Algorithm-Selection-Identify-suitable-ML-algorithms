# 🚀 Deployment Guide: GitHub → Streamlit Community Cloud

This is a beginner-friendly, step-by-step guide since you're new to Git and
deployment. Follow it in order.

## Part 1 — Upload the project to GitHub

### Option A: Using the GitHub website (no command line needed)
1. Go to [github.com](https://github.com) and log in (create a free account if needed).
2. Click the **+** icon (top right) → **New repository**.
3. Name it e.g. `scam-message-detector`, keep it **Public**, do NOT initialize
   with a README (you already have one), then click **Create repository**.
4. On the next page, click **"uploading an existing file"**.
5. Drag and drop **all files and folders** from this project
   (`app.py`, `train_model.py`, `requirements.txt`, `README.md`,
   `.gitignore`, the `data/` folder, and the `model/` folder).
6. Scroll down, write a commit message like "Initial commit", and click
   **Commit changes**.

### Option B: Using Git command line
```bash
cd scam_detector
git init
git add .
git commit -m "Initial commit: AI-Powered Scam Message Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/scam-message-detector.git
git push -u origin main
```
(Replace `YOUR_USERNAME` with your GitHub username, and create the empty repo
on GitHub first as in Option A, steps 1–3.)

> ⚠️ **Important:** Make sure the `model/` folder (containing `model.pkl`,
> `vectorizer.pkl`, `metrics.json`) is actually uploaded/committed. Without
> it, the deployed app has nothing to load and will crash.

## Part 2 — Deploy to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with your GitHub account and authorize Streamlit to access your repos.
3. Click **"New app"**.
4. Choose:
   - **Repository:** `YOUR_USERNAME/scam-message-detector`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. Wait 1–3 minutes while Streamlit installs everything from
   `requirements.txt` and starts your app.
7. You'll get a public URL like:
   `https://scam-message-detector-yourname.streamlit.app`
   — this is the link you send to your teacher.

## Part 3 — If something goes wrong

| Problem | Likely cause | Fix |
|---|---|---|
| "ModuleNotFoundError" on deploy | A package is missing from `requirements.txt` | Add the missing package name and push again |
| App crashes loading `model.pkl` | `model/` folder wasn't uploaded, or scikit-learn version mismatch | Re-check the folder was committed; keep `requirements.txt` versions matching what you trained with |
| App shows old version after you push a fix | Cache | On Streamlit Cloud, click the "⋮" menu → **Reboot app** |
| Push rejected / permission denied (command line) | Not authenticated | Use Option A (website upload) instead, or set up a GitHub Personal Access Token |

## Part 4 — Updating the app later

Any time you push new commits to the `main` branch on GitHub, Streamlit
Cloud automatically redeploys the app with your changes within a minute or two.
