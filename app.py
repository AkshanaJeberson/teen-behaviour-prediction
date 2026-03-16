import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# PAGE CONFIG
st.set_page_config(
    page_title="Viral Social Media Trends Dashboard",
    layout="wide"
)

# -------------------------
# CUSTOM STYLE (MINIMAL)
# -------------------------
st.markdown("""
<style>

body {
    background-color: #f5f5f5;
}

h1, h2, h3 {
    color: #333333;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title("Teen Behavioural Indicators from Viral Social Media Trends")

st.write("""
This dashboard presents the machine learning analysis used to identify
factors that influence viral social media trends and how these trends
relate to behavioural indicators among teenagers.
""")

# -------------------------
# LOAD DATASET
# -------------------------
data = pd.read_csv(r"C:\Users\Akshana\OneDrive\Documents\social_media_viral_content_dataset.csv")

# -------------------------
# MODEL PERFORMANCE
# -------------------------
st.header("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Logistic Regression Accuracy", "0.995")
col2.metric("Decision Tree Accuracy", "1.00")

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.header("Feature Importance")

features = [
    "views",
    "likes",
    "comments",
    "shares",
    "sentiment_score"
]

importance = [0.66, 0.68, 0.41, 0.52, 0.01]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

fig = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale="Greys"
)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# CORRELATION HEATMAP
# -------------------------
st.header("Feature Correlation Heatmap")

numeric_data = data.select_dtypes(include=["int64","float64"])

fig2, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="Greys",
    ax=ax
)

st.pyplot(fig2)

# -------------------------
# CONFUSION MATRIX
# -------------------------
st.header("Model Prediction Breakdown")

cm = [[123,0],
      [2,275]]

fig3, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greys",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig3)

# -------------------------
# DATASET PREVIEW
# -------------------------
st.header("Dataset Preview")

st.dataframe(data.head())

# -------------------------
# BASIC DATA ANALYSIS
# -------------------------
st.header("Dataset Insights")

st.write("Total Posts:", len(data))
st.write("Average Views:", int(data["views"].mean()))
st.write("Average Likes:", int(data["likes"].mean()))
st.write("Average Shares:", int(data["shares"].mean()))

# -------------------------
# TEEN BEHAVIOURAL INDICATORS
# -------------------------
st.header("Teen Behavioural Indicators")

st.write("""
**Engagement Amplification Behaviour**  
High engagement metrics such as likes and views increase the probability of content becoming viral.

**Trend Participation Behaviour**  
Teenagers often replicate trending hashtags, formats, and topics.

**Social Validation Behaviour**  
Likes, shares, and comments act as indicators of peer approval.

**Emotional Engagement Behaviour**  
Emotionally engaging content tends to generate higher interaction.

**High Consumption Behaviour**  
Viral trends increase exposure and content consumption among teenagers.
""")
