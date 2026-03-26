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
# MODERN CSS (REAL PRODUCT STYLE)
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617;
}

/* Cards */
.card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

/* Text */
.small-text {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------

st.sidebar.title("📊 Menu")

page = st.sidebar.radio("", [
    "Home",
    "Prediction",
    "Insights"
])

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
# HOME PAGE
# ---------------------------------

if page == "Home":

    st.markdown("""
    <h1>📱 Teen Behaviour Monitor</h1>
    <p class="small-text">
    Understand how social media trends influence behaviour in a simple and clear way.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="card">
        <h4>Avg Screen Time</h4>
        <h2>{df["Screen Time"].mean():.1f} hrs</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
        <h4>Trend Influence</h4>
        <h2>{"High" if df["Trend Exposure"].mean() > 5 else "Moderate"}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
        <h4>Behaviour Pattern</h4>
        <h2>Moderate</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>🧠 Quick Insight</h3>
    <p class="small-text">
    Social media trends have a noticeable influence on behaviour patterns.
    Balanced usage helps maintain better outcomes.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# PREDICTION PAGE
# ---------------------------------

elif page == "Prediction":

    st.markdown("<h2>🔍 Behaviour Check</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        screen_input = st.slider("⏱ Screen Time (hours/day)", 0.0, 10.0, 4.0)
        interact_input = st.slider("💬 Activity Level", 0, 1000, 200)

    with col2:
        trend_input = st.slider("🔥 Trend Exposure", 0.0, 10.0, 5.0)
        sentiment_input = st.slider("😊 Mood", -1.0, 1.0, 0.0)

    st.markdown("<br>", unsafe_allow_html=True)

    analyze = st.button("✨ Analyze Behaviour")

    if analyze:

        user = np.array([[screen_input, interact_input, trend_input, sentiment_input]])

        prob = model.predict_proba(user)[0][1]
        pred = model.predict(user)[0]

        st.markdown("### 🧠 Result")

        if pred == 1:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#ef4444;">High Influence</h2>
                <p>Strong impact from social media trends.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card">
                <h2 style="color:#22c55e;">Low Influence</h2>
                <p>Balanced behaviour with minimal trend impact.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Influence Level")
        st.progress(prob)

# ---------------------------------
# INSIGHTS PAGE
# ---------------------------------

elif page == "Insights":

    st.markdown("<h2>📊 Behaviour Insights</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    fig1 = px.histogram(df, x="Screen Time", title="Screen Time Pattern",
                        color_discrete_sequence=["#22c55e"])

    fig1.update_layout(
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="white")
    )

    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(df, names="Risk", title="Behaviour Distribution")

    fig2.update_layout(
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="white")
    )

    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <p class="small-text">
    Most users show moderate to high influence from social media trends.
    </p>
    """, unsafe_allow_html=True)

    with st.expander("View Data"):
        st.dataframe(df.head())
