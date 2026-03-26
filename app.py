import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("🌍 EpiPredict AI - Disease Spread Predictor")

st.write("This app predicts future COVID-19 cases using historical data.")

# Load dataset
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# Process data
df = df.iloc[:, 4:].sum()
df = df.reset_index()
df.columns = ["Date", "Cases"]
df["Date"] = pd.to_datetime(df["Date"])

# Show data
st.subheader("📊 COVID-19 Cases Over Time")
st.line_chart(df.set_index("Date"))

# Prepare data
df["Days"] = np.arange(len(df))
X = df[["Days"]]
y = df["Cases"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
days_to_predict = st.slider("Select number of days to predict:", 1, 30, 7)

future_days = np.arange(len(df), len(df)+days_to_predict).reshape(-1, 1)
predictions = model.predict(future_days)

st.subheader("🔮 Predicted Cases")
st.write(predictions)

st.success("Model trained successfully!")