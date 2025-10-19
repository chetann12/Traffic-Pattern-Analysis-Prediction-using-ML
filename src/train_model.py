# app_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------
# Load Model & Feature Columns
# -------------------------
model = joblib.load('../models/random_forest_model.pkl')
feature_columns = joblib.load('../models/training_columns.pkl')

st.set_page_config(page_title="🚦 Traffic Pattern Prediction", layout="centered")

st.title("🚦 Traffic Pattern Analysis & Prediction")
st.write("Designed for Traffic Police to predict congestion levels using ML")

# -------------------------
# User Inputs
# -------------------------
st.subheader("Enter Weather & Time Details")

humidity = st.number_input("Humidity (%)", 0, 100, 70)
temperature = st.number_input("Temperature (°C)", -10, 50, 25)
wind_speed = st.number_input("Wind Speed (km/h)", 0, 100, 5)
visibility = st.number_input("Visibility (miles)", 0, 20, 10)

hour = st.slider("Hour of Day (0–23)", 0, 23, 8)
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
month = st.slider("Month (1–12)", 1, 12, 10)
year = st.number_input("Year", 2012, 2030, 2025)

is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0

# Weather type
weather_type = st.selectbox(
    "Weather Type",
    ["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist", "Drizzle", "Thunderstorm"]
)

# Holiday type
holiday = st.selectbox(
    "Holiday Type",
    ["None", "New Years Day", "Memorial Day", "Independence Day",
     "Labor Day", "Thanksgiving Day", "Columbus Day", "Veterans Day"]
)

# -------------------------
# Prepare Input Data
# -------------------------
input_data = {
    "humidity": humidity,
    "wind_speed": wind_speed,
    "visibility_in_miles": visibility,
    "temperature": temperature,
    "hour": hour,
    "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
    "month": month,
    "year": year,
    "is_weekend": is_weekend,
    "rush_hour": rush_hour,
    "traffic_prev_hour": 0,
    "traffic_prev_day_same_hour": 0,
    "temp_humidity": temperature * humidity
}

# Add one-hot encodings dynamically
for wt in ["Clouds", "Drizzle", "Fog", "Haze", "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]:
    input_data[f"weather_type_{wt}"] = 1 if wt == weather_type else 0

for hol in ["Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day", "Memorial Day",
             "New Years Day", "None", "State Fair", "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"]:
    input_data[f"is_holiday_{hol}"] = 1 if hol == holiday else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Reindex to match training columns
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -------------------------
# Predict Button
# -------------------------
if st.button("🔍 Predict Traffic Level"):
    pred_log = model.predict(input_df)[0]
    prediction = np.expm1(pred_log)  # Convert log back to actual scale

    st.metric(label="Predicted Traffic Volume", value=f"{prediction:.0f} vehicles/hour")

    if prediction > 4000:
        st.error("🚨 High Traffic Expected – Deploy Additional Units")
    elif prediction > 2000:
        st.warning("⚠️ Moderate Traffic Expected – Stay Alert")
    else:
        st.success("🟢 Low Traffic Expected – Normal Flow")

