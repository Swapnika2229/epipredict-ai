import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="EpiPredict AI", page_icon="🌍", layout="wide")

# 🔥 Custom CSS (FINAL UI)
st.markdown("""
<style>
/* Sidebar full background */
section[data-testid="stSidebar"] {
    background-color: #0e1117;
}

/* Selectbox container (closed state) */
div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: white !important;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown menu (opened list) */
ul[role="listbox"] {
    background-color: #1f2937 !important;
}

/* Dropdown options */
li[role="option"] {
    background-color: #1f2937 !important;
    color: white !important;
}

/* Hover effect */
li[role="option"]:hover {
    background-color: #374151 !important;
}

/* Label (Select Country) */
label {
    color: #00e6e6 !important;
    font-weight: bold;
}
            
/* Main background */
body {
    background-color: #0e1117;
}

/* Title colors */
h1, h2, h3 {
    color: #00e6e6;
}

/* 🔥 FIX METRIC CARDS */
[data-testid="stMetric"] {
    background-color: #1f2937 !important;
    padding: 15px;
    border-radius: 10px;
    color: white !important;
}

/* 🔥 FIX METRIC TEXT */
[data-testid="stMetric"] label {
    color: #9ca3af !important;
}
[data-testid="stMetric"] div {
    color: #ffffff !important;
}

/* 🔥 FIX SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Dropdown (country select) */
.stSelectbox div {
    color: black !important;
    background-color: white !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("🌍 EpiPredict AI - Disease Spread Predictor")
st.write("### 🚀 AI-Powered Epidemic Forecasting Dashboard")

# Load dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Sidebar
st.sidebar.header("⚙️ Controls")

countries = df["Country/Region"].unique()
selected_country = st.sidebar.selectbox("🌍 Select Country", countries)

days_to_predict = st.sidebar.slider("📅 Days to Predict", 1, 30, 7)

# Process selected country
country_df = df[df["Country/Region"] == selected_country]
country_df = country_df.iloc[:, 4:].sum()
country_df = country_df.reset_index()
country_df.columns = ["Date", "Cases"]
country_df["Date"] = pd.to_datetime(country_df["Date"])

# Metrics
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Cases", int(country_df["Cases"].iloc[-1]))
col2.metric("Days Recorded", len(country_df))
col3.metric("Avg Daily Cases", int(country_df["Cases"].mean()))

# ML Model
country_df["Days"] = np.arange(len(country_df))
X = country_df[["Days"]]
y = country_df["Cases"]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(country_df), len(country_df)+days_to_predict).reshape(-1, 1)
predictions = model.predict(future_days)

# Graph 1: Trend
st.markdown("##")
st.subheader("📈 Actual vs Predicted Trend")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(country_df["Date"], country_df["Cases"], label="Actual", linewidth=2)

future_dates = pd.date_range(start=country_df["Date"].iloc[-1], periods=days_to_predict+1)[1:]
ax.plot(future_dates, predictions, linestyle="dashed", linewidth=2, label="Predicted")

ax.legend()
ax.set_title(f"{selected_country} Trend")

st.pyplot(fig, use_container_width=True)

# Graph 2: Bar Chart
st.markdown("##")
st.subheader("📊 Prediction Breakdown")

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.bar(range(1, days_to_predict+1), predictions)
ax2.set_xlabel("Days")
ax2.set_ylabel("Predicted Cases")
ax2.set_title("Future Cases (Bar View)")

st.pyplot(fig2, use_container_width=True)

# Top countries analysis
st.markdown("##")
st.subheader("🌍 Top 5 Countries by Cases")

top_countries = df.groupby("Country/Region").sum().iloc[:, -1].sort_values(ascending=False).head(5)

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.bar(top_countries.index, top_countries.values)
ax3.set_title("Top 5 Countries")

st.pyplot(fig3, use_container_width=True)

# Predictions
st.subheader("🔮 Future Predictions")

for i, val in enumerate(predictions):
    st.write(f"Day {i+1}: {int(val)} cases")

# Smart Insights
st.subheader("💡 Insights")

growth = predictions[-1] - predictions[0]

if growth > 0:
    st.warning("Cases are expected to rise significantly 📈")
else:
    st.success("Cases are stabilizing or decreasing 📉")

st.info("This tool helps in planning healthcare resources and early intervention strategies.")

# Footer
st.markdown("---")
st.write("🚀 Developed by Solo Innovator | Codecure Hackathon")
