import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Teen Behaviour Prediction", layout="wide")

st.title("📊 Teen Behaviour Prediction Dashboard")
st.subheader("Predictive Analysis using Logistic Regression")

# -----------------------------------------
# Generate Synthetic Dataset
# -----------------------------------------

np.random.seed(42)
data_size = 500

screen_time = np.random.uniform(1, 8, data_size)
interactions = np.random.randint(10, 500, data_size)
trend_exposure = np.random.uniform(0, 10, data_size)
sentiment_score = np.random.uniform(-1, 1, data_size)

risk = (screen_time * 0.4 + trend_exposure * 0.3 + interactions * 0.002 > 3).astype(int)

df = pd.DataFrame({
    "ScreenTime": screen_time,
    "Interactions": interactions,
    "TrendExposure": trend_exposure,
    "Sentiment": sentiment_score,
    "Risk": risk
})

# -----------------------------------------
# Train Logistic Regression
# -----------------------------------------

X = df[["ScreenTime", "Interactions", "TrendExposure", "Sentiment"]]
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Sidebar Model Performance
st.sidebar.header("📈 Model Performance")
st.sidebar.write(f"Accuracy: {round(accuracy*100,2)}%")

# -----------------------------------------
# User Input Section
# -----------------------------------------

st.header("🔍 Predict Teen Behaviour Risk")

col1, col2 = st.columns(2)

with col1:
    user_screen_time = st.slider("Daily Screen Time (hours)", 0.0, 10.0, 4.0)
    user_interactions = st.slider("Daily Interactions", 0, 1000, 200)

with col2:
    user_trend = st.slider("Trend Exposure Score (0-10)", 0.0, 10.0, 5.0)
    user_sentiment = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.0)

if st.button("Predict Behaviour Risk"):

    user_data = np.array([[user_screen_time,
                           user_interactions,
                           user_trend,
                           user_sentiment]])

    probability = model.predict_proba(user_data)[0][1]
    prediction = model.predict(user_data)[0]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk Behaviour\nProbability: {round(probability*100,2)}%")
    else:
        st.success(f"✅ Low Risk Behaviour\nProbability: {round(probability*100,2)}%")

# -----------------------------------------
# Confusion Matrix
# -----------------------------------------

st.header("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# -----------------------------------------
# ROC Curve
# -----------------------------------------

st.header("📈 ROC Curve")

y_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax_roc.plot([0,1],[0,1],'--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# -----------------------------------------
# Feature Importance (Coefficients)
# -----------------------------------------

st.header("🧠 Logistic Regression Coefficients")

coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

st.dataframe(coeff_df)

# -----------------------------------------
# Dataset Preview
# -----------------------------------------

with st.expander("📂 View Sample Dataset"):
    st.dataframe(df.head())