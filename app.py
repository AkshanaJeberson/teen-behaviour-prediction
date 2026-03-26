import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(page_title="Teen Behaviour Dashboard", layout="wide")

# ---------------------------------
# CUSTOM CSS (DASHBOARD STYLE)
# ---------------------------------

st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.stApp {
    background-color: #0f172a;
    color: white;
}
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0;
}
h1, h2, h3, h4 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HEADER
# ---------------------------------

st.title("📊 Teen Behaviour Analytics Dashboard")
st.markdown("Understand how social media trends influence teen behaviour using AI.")

# ---------------------------------
# DATA GENERATION
# ---------------------------------

np.random.seed(42)
data_size = 500

screen_time = np.random.uniform(1, 8, data_size)
interactions = np.random.randint(10, 500, data_size)
trend_exposure = np.random.uniform(0, 10, data_size)
sentiment = np.random.uniform(-1, 1, data_size)

risk = (screen_time*0.4 + trend_exposure*0.3 + interactions*0.002 > 3).astype(int)

df = pd.DataFrame({
    "Screen Time": screen_time,
    "Interactions": interactions,
    "Trend Exposure": trend_exposure,
    "Sentiment": sentiment,
    "Risk": risk
})

# ---------------------------------
# MODEL
# ---------------------------------

X = df[["Screen Time", "Interactions", "Trend Exposure", "Sentiment"]]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# ---------------------------------
# DASHBOARD CARDS
# ---------------------------------

st.subheader("Dashboard Summary")

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f'<div class="card"><h4>Total Users</h4><h2>{len(df)}</h2></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="card"><h4>Avg Screen Time</h4><h2>{df["Screen Time"].mean():.1f} hrs</h2></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="card"><h4>High Risk Cases</h4><h2>{df["Risk"].sum()}</h2></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="card"><h4>Model Accuracy</h4><h2>{accuracy*100:.1f}%</h2></div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------
# PREDICTION PANEL
# ---------------------------------

st.subheader("🔍 Behaviour Prediction")

col1, col2 = st.columns(2)

with col1:
    screen_input = st.slider("Daily Screen Time (hours)", 0.0, 10.0, 4.0)
    interact_input = st.slider("Daily Interactions", 0, 1000, 200)

with col2:
    trend_input = st.slider("Trend Exposure", 0.0, 10.0, 5.0)
    sentiment_input = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

if st.button("Predict Behaviour Risk"):

    user = np.array([[screen_input, interact_input, trend_input, sentiment_input]])

    prob = model.predict_proba(user)[0][1]
    pred = model.predict(user)[0]

    st.subheader("Prediction Result")

    if pred == 1:
        st.markdown(f'<div class="card" style="color:#ef4444;"><h2>⚠ High Risk</h2><p>Probability: {prob*100:.1f}%</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card" style="color:#22c55e;"><h2>✅ Low Risk</h2><p>Probability: {(1-prob)*100:.1f}%</p></div>', unsafe_allow_html=True)

    st.progress(prob)

st.divider()

# ---------------------------------
# INSIGHTS (USER FRIENDLY)
# ---------------------------------

st.subheader("Behaviour Insights")

col1, col2 = st.columns(2)

fig1 = px.histogram(df, x="Screen Time", title="Screen Time Distribution", color_discrete_sequence=["#22c55e"])
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(df, names="Risk", title="Risk Distribution")
col2.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------
# MODEL INSIGHTS (BOTTOM)
# ---------------------------------

st.subheader("Model Insights")

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])

st.write("Confusion Matrix")
st.dataframe(cm_df)

# ROC Curve
y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr
})

fig3 = px.line(roc_df, x="FPR", y="TPR", title="ROC Curve")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------
# DATASET
# ---------------------------------

with st.expander("📂 View Dataset"):
    st.dataframe(df.head())
