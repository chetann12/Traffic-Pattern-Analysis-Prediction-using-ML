import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, time

# -------------------------------
# Load model and training columns
# -------------------------------
model_path = "../models/random_forest_model.pkl"
columns_path = "../models/training_columns.pkl"
model = joblib.load(model_path)
training_columns = joblib.load(columns_path)

st.set_page_config(page_title="🚦Traffic Prediction Dashboard", layout="wide")

st.title("🚦 Smart Traffic Prediction Dashboard for Police")
st.markdown("Predict traffic congestion based on weather, date, and time conditions.")

# -------------------------------
# Input Section
# -------------------------------
st.header("📅 Select Date & Time")
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Select Date", date.today())
with col2:
    selected_time = st.time_input("Select Time", time(8, 0))

# Combine into datetime
selected_datetime = datetime.combine(selected_date, selected_time)
hour = selected_datetime.hour
day_of_week = selected_datetime.weekday()  # 0=Mon
month = selected_datetime.month
year = selected_datetime.year
is_weekend = 1 if day_of_week >= 5 else 0

# -------------------------------
# Weather Inputs
# -------------------------------
st.header("🌦 Weather Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    wind_speed = st.slider("Wind Speed (mph)", 0, 30, 5)
with col2:
    visibility = st.slider("Visibility (miles)", 0, 20, 10)
    temperature = st.slider("Temperature (°F)", -10, 120, 75)
with col3:
    weather_type = st.selectbox(
        "Weather Type",
        ["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist", "Thunderstorm"]
    )
    weather_description = st.selectbox(
        "Weather Description",
        ["Sky is Clear", "LightRain", "HeavyRain", "Fog", "Snow", "Thunderstorm"]
    )

# -------------------------------
# Auto Holiday Detection
# -------------------------------
us_holidays = {
    "New Years Day": (1, 1),
    "Independence Day": (7, 4),
    "Labor Day": (9, 2),
    "Thanksgiving Day": (11, 28),
    "Christmas Day": (12, 25)
}
holiday_found = None
for name, (m, d) in us_holidays.items():
    if selected_date.month == m and selected_date.day == d:
        holiday_found = name
        break
is_holiday_cols = {f"is_holiday_{h}": 0 for h in [
    "Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day",
    "Memorial Day", "New Years Day", "None", "State Fair",
    "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
]}
if holiday_found:
    is_holiday_cols[f"is_holiday_{holiday_found}"] = 1
else:
    is_holiday_cols["is_holiday_None"] = 1

# -------------------------------
# Derived Features
# -------------------------------
rush_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
temp_humidity = temperature * humidity

# Auto placeholders for unavailable temporal data
traffic_prev_hour = 0
traffic_prev_day_same_hour = 0

# -------------------------------
# Construct input DataFrame
# -------------------------------
input_data = {
    "humidity": humidity,
    "wind_speed": wind_speed,
    "visibility_in_miles": visibility,
    "temperature": temperature,
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "year": year,
    "is_weekend": is_weekend,
    "rush_hour": rush_hour,
    "traffic_prev_hour": traffic_prev_hour,
    "traffic_prev_day_same_hour": traffic_prev_day_same_hour,
    "temp_humidity": temp_humidity
}

# Add weather type and description (one-hot encoding)
weather_columns = [
    'weather_description_Fog', 'weather_description_HeavyRain', 'weather_description_LightRain',
    'weather_description_Sky is Clear', 'weather_description_Snow',
    'weather_description_Squalls', 'weather_description_Thunderstorm',
    'weather_type_Clouds', 'weather_type_Drizzle', 'weather_type_Fog',
    'weather_type_Haze', 'weather_type_Mist', 'weather_type_Rain',
    'weather_type_Smoke', 'weather_type_Snow', 'weather_type_Squall', 'weather_type_Thunderstorm'
]
for col in weather_columns:
    input_data[col] = 1 if (weather_description in col or weather_type in col) else 0

# Add holiday columns
input_data.update(is_holiday_cols)

# Align with model columns
input_df = pd.DataFrame([input_data])
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[training_columns]

# -------------------------------
# Predict Button
# -------------------------------
st.header("🔮 Predict Traffic Volume")
if st.button("🚗 Predict Traffic"):
    prediction = np.expm1(model.predict(input_df)[0])
    st.success(f"Predicted Traffic Volume: **{prediction:.0f} vehicles/hour**")

    if prediction > 4000:
        st.error("⚠️ Heavy Traffic Expected! Consider deploying extra officers.")
    elif prediction > 2000:
        st.warning("🟠 Moderate Traffic. Be prepared.")
    else:
        st.info("🟢 Low Traffic. Normal conditions expected.")
