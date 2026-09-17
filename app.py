"""
app.py
------
AI-Powered Scam Message Detector — Streamlit Dashboard

Loads the trained TF-IDF vectorizer + ML model (produced by train_model.py)
and provides an interactive UI to analyze SMS/WhatsApp-style messages for
scam risk, with confidence scores, a 0-100 risk meter, risk indicators,
example messages, and a session-based prediction history.
"""

import re
import json
from datetime import datetime

import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Scam Message Detector",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model artifacts (cached so they load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    with open("model/metrics.json") as f:
        metrics = json.load(f)
    return model, vectorizer, metrics

model, vectorizer, metrics = load_artifacts()

# ---------------------------------------------------------------------------
# Risk-indicator patterns (used only for the explanatory "why" section —
# the actual classification always comes from the trained ML model)
# ---------------------------------------------------------------------------
RISK_PATTERNS = {
    "Urgency language": r"\b(urgent|immediately|now|act fast|final notice|last chance|expires?|hurry|limited time)\b",
    "Money / prize mention": r"\b(won|winner|prize|free|cash|reward|inheritance|refund|cashback|lottery)\b",
    "Suspicious link": r"(https?://|www\.|bit\.ly|tinyurl|\.com|\.net|\.info|\.co\b)",
    "Requests personal/financial info": r"\b(verify|password|otp|pin|card number|account number|ssn|bank details|confirm your)\b",
    "Threat of account/legal action": r"\b(suspend|suspended|closed|locked|arrest|penalty|legal action|blocked)\b",
    "Unusual payment request": r"\b(send money|wire transfer|gift card|western union|customs fee|processing fee)\b",
}

def find_risk_indicators(text: str):
    text_lower = text.lower()
    found = []
    for label, pattern in RISK_PATTERNS.items():
        if re.search(pattern, text_lower):
            found.append(label)
    return found

def compute_risk_score(label: str, confidence: float, num_indicators: int) -> int:
    """
    Combine model confidence + rule-based indicator count into an
    intuitive 0-100 risk score for display purposes.
    """
    base = {"SAFE": 5, "SUSPICIOUS": 45, "SCAM": 75}.get(label, 50)
    score = base + confidence * 20 + min(num_indicators, 5) * 3
    return int(max(0, min(100, round(score))))

def risk_level(score: int) -> str:
    if score < 34:
        return "LOW"
    elif score < 67:
        return "MEDIUM"
    else:
        return "HIGH"

RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
LABEL_EMOJI = {"SAFE": "✅", "SUSPICIOUS": "⚠️", "SCAM": "🚨"}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "message_input" not in st.session_state:
    st.session_state.message_input = ""

EXAMPLES = {
    "🎁 Prize scam": "Congratulations! You have WON $5000! Click bit.ly/claim-now to claim your prize before it expires.",
    "🏦 Fake bank alert": "URGENT: Your bank account has been suspended. Verify your details immediately at secure-verify-account.com or lose access.",
    "⚠️ Suspicious offer": "Hi, I saw your profile and think you'd be a great fit for a part-time role. Interested? Reply for details.",
    "✅ Normal message": "Hey, are we still meeting for lunch at 1 PM tomorrow?",
}

def set_example(text):
    st.session_state.message_input = text

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛡️ AI-Powered Scam Message Detector")
st.caption("Paste an SMS or WhatsApp message below and let the ML model analyze it for scam risk.")

with st.expander("ℹ️ About this model"):
    # Support both metrics.json schemas: the Day 2 single-split format
    # ({"results": {...}}) and the Day 3 cross-validation finalized format
    # ({"holdout_split_metrics": {...}}).
    if "holdout_split_metrics" in metrics:
        m = metrics["holdout_split_metrics"]
        selection_note = metrics.get("selection_method", "train/test split")
    else:
        m = metrics["results"][metrics["best_model"]]
        selection_note = "train/test split"

    st.write(
        f"**Best performing model:** {metrics['best_model']}  \n"
        f"**Selected via:** {selection_note}  \n"
        f"**Test accuracy:** {m['accuracy']*100:.1f}%  \n"
        f"**F1 score:** {m['f1_score']*100:.1f}%  \n"
        f"Trained on {metrics['train_size']} messages, tested on {metrics['test_size']} messages, "
        f"comparing Logistic Regression, Multinomial Naive Bayes, and Linear SVM."
    )

st.divider()

# ---------------------------------------------------------------------------
# Example buttons
# ---------------------------------------------------------------------------
st.subheader("Try an example")
cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(cols, EXAMPLES.items()):
    with col:
        st.button(label, on_click=set_example, args=(text,), use_container_width=True)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
message = st.text_area(
    "Message to analyze",
    key="message_input",
    height=120,
    placeholder="Paste the SMS or WhatsApp message here...",
)

col_a, col_b = st.columns([1, 5])
with col_a:
    analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        vec = vectorizer.transform([message])
        pred_label = model.predict(vec)[0]

        # Confidence — use predict_proba when available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            classes = model.classes_
            confidence = float(max(proba))
            proba_dict = dict(zip(classes, proba))
        else:
            confidence = 1.0
            proba_dict = {pred_label: 1.0}

        indicators = find_risk_indicators(message)
        score = compute_risk_score(pred_label, confidence, len(indicators))
        level = risk_level(score)

        st.session_state.history.insert(
            0,
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "label": pred_label,
                "confidence": confidence,
                "score": score,
                "level": level,
            },
        )

        st.divider()
        st.subheader("Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Classification", f"{LABEL_EMOJI.get(pred_label, '')} {pred_label}")
        with r2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with r3:
            st.metric("Risk Score", f"{score}/100", help="0 = very safe, 100 = very high risk")

        # Risk meter
        st.markdown(
            f"""
            <div style="background:#eee;border-radius:8px;height:22px;width:100%;overflow:hidden;">
              <div style="background:{RISK_COLORS[level]};width:{score}%;height:100%;
                          display:flex;align-items:center;justify-content:flex-end;
                          padding-right:8px;color:white;font-weight:bold;font-size:12px;">
                {level} RISK
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        if pred_label == "SCAM":
            st.error("🚨 This message shows strong characteristics of a scam. Do not click any links or share personal information.")
        elif pred_label == "SUSPICIOUS":
            st.warning("⚠️ This message has some suspicious characteristics. Proceed with caution and verify the sender independently.")
        else:
            st.success("✅ This message looks safe based on the patterns the model has learned.")

        # Class probability breakdown
        with st.expander("📊 Full probability breakdown"):
            for cls in sorted(proba_dict.keys()):
                st.write(f"**{cls}**")
                st.progress(float(proba_dict[cls]))

        # Risk indicators found
        st.subheader("🔎 Risk Indicators Detected")
        if indicators:
            for ind in indicators:
                st.markdown(f"- ⚠️ {ind}")
        else:
            st.markdown("- No common scam-language patterns detected.")

        # Message stats
        st.subheader("📈 Message Stats")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Characters", len(message))
        s2.metric("Words", len(message.split()))
        s3.metric("Links found", len(re.findall(r"(https?://|www\.|bit\.ly|tinyurl)", message.lower())))
        s4.metric("Exclamation marks", message.count("!"))

st.divider()

# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------
st.subheader("🕓 Prediction History (this session)")
if st.session_state.history:
    if st.button("🗑️ Clear history"):
        st.session_state.history = []
        st.rerun()

    for entry in st.session_state.history:
        emoji = LABEL_EMOJI.get(entry["label"], "")
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"**{emoji} {entry['label']}** — _{entry['time']}_")
                st.caption(entry["message"][:140] + ("..." if len(entry["message"]) > 140 else ""))
            with c2:
                st.write(f"Risk: **{entry['score']}/100**")
                st.caption(entry["level"])
else:
    st.caption("No messages analyzed yet in this session.")

st.divider()
st.caption("Built for a university semester project - ML: TF-IDF + " + metrics["best_model"] + " - Not a substitute for official fraud verification.")
