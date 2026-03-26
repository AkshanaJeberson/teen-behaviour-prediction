import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(page_title="Teen Behaviour Monitor", layout="wide")

# ---------------------------------
# SIMPLE CLEAN STYLE
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 10px;
}
.small-text {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# HEADER
# ---------------------------------

st.title("📱 Teen Behaviour Monitor")
st.markdown('<p class="small-text">Simple insights on how social media trends influence behaviour</p>', unsafe_allow_html=True)

st.divider()

# ---------------------------------
# DATA + MODEL
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

X = df[["Screen Time", "Interactions", "Trend Exposure", "Sentiment"]]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------------
# SIMPLE SUMMARY
# ---------------------------------

st.subheader("Overview")

col1, col2, col3 = st.columns(3)

col1.markdown(f'<div class="card"><h4>Avg Screen Time</h4><h2>{df["Screen Time"].mean():.1f} hrs</h2></div>', unsafe_allow_html=True)

col2.markdown(f'<div class="card"><h4>Trend Influence</h4><h2>{"High" if df["Trend Exposure"].mean() > 5 else "Moderate"}</h2></div>', unsafe_allow_html=True)

col3.markdown(f'<div class="card"><h4>Behaviour Level</h4><h2>{"Moderate"}</h2></div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------
# INPUT SECTION
# ---------------------------------

st.subheader("🔍 Check Behaviour")

col1, col2 = st.columns(2)

with col1:
    screen_input = st.slider("Screen Time (hours/day)", 0.0, 10.0, 4.0)
    interact_input = st.slider("Social Media Activity", 0, 1000, 200)

with col2:
    trend_input = st.slider("Trend Exposure", 0.0, 10.0, 5.0)
    sentiment_input = st.slider("Mood / Sentiment", -1.0, 1.0, 0.0)

# ---------------------------------
# PREDICTION
# ---------------------------------

if st.button("Analyze Behaviour"):

    user = np.array([[screen_input, interact_input, trend_input, sentiment_input]])

    prob = model.predict_proba(user)[0][1]
    pred = model.predict(user)[0]

    st.subheader("Result")

    if pred == 1:
        st.markdown(
            f'<div class="card"><h3 style="color:#ef4444;">High Influence</h3><p>Social media trends are strongly affecting behaviour.</p></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="card"><h3 style="color:#22c55e;">Low Influence</h3><p>Behaviour is less influenced by social media trends.</p></div>',
            unsafe_allow_html=True
        )

    st.progress(prob)

st.divider()

# ---------------------------------
# INSIGHTS
# ---------------------------------

st.subheader("Insights")

col1, col2 = st.columns(2)

fig1 = px.histogram(df, x="Screen Time", title="Screen Time Pattern", color_discrete_sequence=["#22c55e"])
col1.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(df, names="Risk", title="Behaviour Distribution")
col2.plotly_chart(fig2, use_container_width=True)

st.markdown('<p class="small-text">Most users show moderate to high influence from trends.</p>', unsafe_allow_html=True)

st.divider()

# ---------------------------------
# OPTIONAL DATA
# ---------------------------------

with st.expander("View Data"):
    st.dataframe(df.head())
