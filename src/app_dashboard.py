import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Load trained model and metadata
# -------------------------------
MODEL_PATH = "../models/random_forest_model.pkl"
TRAINING_COLUMNS_PATH = "../models/training_columns.pkl"

model = joblib.load(MODEL_PATH)
training_columns = joblib.load(TRAINING_COLUMNS_PATH)

st.set_page_config(page_title="Traffic Pattern Prediction Dashboard", layout="centered")
st.title("🚦 Traffic Pattern Analysis & Prediction Dashboard")

st.markdown("### Enter weather and environmental parameters to predict traffic volume.")

# -------------------------------
# Input UI Components
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    humidity = st.slider("Humidity (%)", 0, 100, 60)
    wind_speed = st.slider("Wind Speed (mph)", 0, 50, 10)
    visibility = st.slider("Visibility (miles)", 0, 20, 10)
    temperature = st.number_input("Temperature (°F)", min_value=-20.0, max_value=120.0, value=70.0)
    traffic_prev_hour = st.number_input("Traffic Previous Hour", min_value=0, value=3000)

with col2:
    traffic_prev_day_same_hour = st.number_input("Traffic Previous Day (Same Hour)", min_value=0, value=2800)
    hour = st.slider("Hour of Day", 0, 23, 8)
    day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
    month = st.slider("Month", 1, 12, 6)
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2025)

is_weekend = 1 if day_of_week in [5, 6] else 0
rush_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0

# Weather dropdowns
st.markdown("### 🌦️ Weather Conditions")
weather_description = st.selectbox(
    "Weather Description",
    ["Fog", "HeavyRain", "LightRain", "Sky is Clear", "Snow", "Squalls", "Thunderstorm"]
)
weather_type = st.selectbox(
    "Weather Type",
    ["Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
)

holiday = st.selectbox(
    "Holiday",
    [
        "None", "Columbus Day", "Independence Day", "Labor Day",
        "Martin Luther King Jr Day", "Memorial Day", "New Years Day",
        "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
    ]
)

# Derived feature
temp_humidity = temperature * humidity

# -------------------------------
# Create input DataFrame
# -------------------------------
input_data = {
    "humidity": [humidity],
    "wind_speed": [wind_speed],
    "visibility_in_miles": [visibility],
    "temperature": [temperature],
    "hour": [hour],
    "day_of_week": [day_of_week],
    "month": [month],
    "year": [year],
    "is_weekend": [is_weekend],
    "rush_hour": [rush_hour],
    "traffic_prev_hour": [traffic_prev_hour],
    "traffic_prev_day_same_hour": [traffic_prev_day_same_hour],
    "temp_humidity": [temp_humidity],
}

# Add one-hot encoded weather/holiday features
for desc in [
    "Fog", "HeavyRain", "LightRain", "Sky is Clear", "Snow", "Squalls", "Thunderstorm"
]:
    input_data[f"weather_description_{desc}"] = [1 if weather_description == desc else 0]

for wtype in [
    "Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
]:
    input_data[f"weather_type_{wtype}"] = [1 if weather_type == wtype else 0]

for hol in [
    "Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day",
    "Memorial Day", "New Years Day", "None", "State Fair", "Thanksgiving Day",
    "Veterans Day", "Washingtons Birthday"
]:
    input_data[f"is_holiday_{hol}"] = [1 if holiday == hol else 0]

# Create dataframe
input_df = pd.DataFrame(input_data)

# -------------------------------
# Align with training columns
# -------------------------------
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match training
input_df = input_df[training_columns]

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🚀 Predict Traffic Volume"):
    prediction = np.expm1(model.predict(input_df)[0])  # inverse log-transform
    st.success(f"Predicted Traffic Volume: **{prediction:.0f} vehicles/hour**")

    if prediction > 5000:
        st.error("🚨 High Traffic Expected — Deploy more traffic police units!")
    elif prediction > 2000:
        st.warning("⚠️ Moderate Traffic Expected — Keep monitoring.")
    else:
        st.info("✅ Low Traffic — Normal conditions.")

