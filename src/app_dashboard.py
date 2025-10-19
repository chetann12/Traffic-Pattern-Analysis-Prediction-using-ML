# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, date, time
#
# # -------------------------------
# # Load model and training columns
# # -------------------------------
# model_path = "../models/random_forest_model.pkl"
# columns_path = "../models/training_columns.pkl"
# model = joblib.load(model_path)
# training_columns = joblib.load(columns_path)
#
# st.set_page_config(page_title="🚦Traffic Prediction Dashboard", layout="wide")
#
# st.title("🚦 Smart Traffic Prediction Dashboard for Police")
# st.markdown("Predict traffic congestion based on weather, date, and time conditions.")
#
# # -------------------------------
# # Input Section
# # -------------------------------
# st.header("📅 Select Date & Time")
# col1, col2 = st.columns(2)
# with col1:
#     selected_date = st.date_input("Select Date", date.today())
# with col2:
#     selected_time = st.time_input("Select Time", time(8, 0))
#
# # Combine into datetime
# selected_datetime = datetime.combine(selected_date, selected_time)
# hour = selected_datetime.hour
# day_of_week = selected_datetime.weekday()  # 0=Mon
# month = selected_datetime.month
# year = selected_datetime.year
# is_weekend = 1 if day_of_week >= 5 else 0
#
# # -------------------------------
# # Weather Inputs
# # -------------------------------
# st.header("🌦 Weather Conditions")
# col1, col2, col3 = st.columns(3)
# with col1:
#     humidity = st.slider("Humidity (%)", 0, 100, 70)
#     wind_speed = st.slider("Wind Speed (mph)", 0, 30, 5)
# with col2:
#     visibility = st.slider("Visibility (miles)", 0, 20, 10)
#     temperature = st.slider("Temperature (°F)", -10, 120, 75)
# with col3:
#     weather_type = st.selectbox(
#         "Weather Type",
#         ["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist", "Thunderstorm"]
#     )
#     weather_description = st.selectbox(
#         "Weather Description",
#         ["Sky is Clear", "LightRain", "HeavyRain", "Fog", "Snow", "Thunderstorm"]
#     )
#
# # -------------------------------
# # Auto Holiday Detection
# # -------------------------------
# us_holidays = {
#     "New Years Day": (1, 1),
#     "Independence Day": (7, 4),
#     "Labor Day": (9, 2),
#     "Thanksgiving Day": (11, 28),
#     "Christmas Day": (12, 25)
# }
# holiday_found = None
# for name, (m, d) in us_holidays.items():
#     if selected_date.month == m and selected_date.day == d:
#         holiday_found = name
#         break
# is_holiday_cols = {f"is_holiday_{h}": 0 for h in [
#     "Columbus Day", "Independence Day", "Labor Day", "Martin Luther King Jr Day",
#     "Memorial Day", "New Years Day", "None", "State Fair",
#     "Thanksgiving Day", "Veterans Day", "Washingtons Birthday"
# ]}
# if holiday_found:
#     is_holiday_cols[f"is_holiday_{holiday_found}"] = 1
# else:
#     is_holiday_cols["is_holiday_None"] = 1
#
# # -------------------------------
# # Derived Features
# # -------------------------------
# rush_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
# temp_humidity = temperature * humidity
#
# # Auto placeholders for unavailable temporal data
# traffic_prev_hour = 0
# traffic_prev_day_same_hour = 0
#
# # -------------------------------
# # Construct input DataFrame
# # -------------------------------
# input_data = {
#     "humidity": humidity,
#     "wind_speed": wind_speed,
#     "visibility_in_miles": visibility,
#     "temperature": temperature,
#     "hour": hour,
#     "day_of_week": day_of_week,
#     "month": month,
#     "year": year,
#     "is_weekend": is_weekend,
#     "rush_hour": rush_hour,
#     "traffic_prev_hour": traffic_prev_hour,
#     "traffic_prev_day_same_hour": traffic_prev_day_same_hour,
#     "temp_humidity": temp_humidity
# }
#
# # Add weather type and description (one-hot encoding)
# weather_columns = [
#     'weather_description_Fog', 'weather_description_HeavyRain', 'weather_description_LightRain',
#     'weather_description_Sky is Clear', 'weather_description_Snow',
#     'weather_description_Squalls', 'weather_description_Thunderstorm',
#     'weather_type_Clouds', 'weather_type_Drizzle', 'weather_type_Fog',
#     'weather_type_Haze', 'weather_type_Mist', 'weather_type_Rain',
#     'weather_type_Smoke', 'weather_type_Snow', 'weather_type_Squall', 'weather_type_Thunderstorm'
# ]
# for col in weather_columns:
#     input_data[col] = 1 if (weather_description in col or weather_type in col) else 0
#
# # Add holiday columns
# input_data.update(is_holiday_cols)
#
# # Align with model columns
# input_df = pd.DataFrame([input_data])
# for col in training_columns:
#     if col not in input_df.columns:
#         input_df[col] = 0
# input_df = input_df[training_columns]
#
# # -------------------------------
# # Predict Button
# # -------------------------------
# st.header("🔮 Predict Traffic Volume")
# if st.button("🚗 Predict Traffic"):
#     prediction = np.expm1(model.predict(input_df)[0])
#     st.success(f"Predicted Traffic Volume: **{prediction:.0f} vehicles/hour**")
#
#     if prediction > 4000:
#         st.error("⚠️ Heavy Traffic Expected! Consider deploying extra officers.")
#     elif prediction > 2000:
#         st.warning("🟠 Moderate Traffic. Be prepared.")
#     else:
#         st.info("🟢 Low Traffic. Normal conditions expected.")
#
#


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
# Enhanced Traffic Condition Prediction
# -------------------------------
st.header("🔮 Traffic Condition Analysis")


def calculate_traffic_condition(predicted_volume, weather_condition, visibility, wind_speed):
    """
    Calculate traffic condition considering both volume and weather impact
    """
    # Base capacity under ideal conditions (vehicles/hour)
    base_capacity = 5000

    # Weather impact factors (reduce effective capacity)
    weather_impact = {
        "Sky is Clear": 1.0,
        "LightRain": 0.7,
        "HeavyRain": 0.5,
        "Fog": 0.4,
        "Snow": 0.3,
        "Thunderstorm": 0.4
    }

    # Visibility impact
    if visibility < 1:
        visibility_factor = 0.4
    elif visibility < 3:
        visibility_factor = 0.6
    elif visibility < 5:
        visibility_factor = 0.8
    else:
        visibility_factor = 1.0

    # Wind speed impact (for high winds)
    wind_factor = 0.8 if wind_speed > 20 else 1.0

    # Calculate effective capacity
    weather_factor = weather_impact.get(weather_description, 0.7)
    effective_capacity = base_capacity * weather_factor * visibility_factor * wind_factor

    # Calculate congestion level
    congestion_ratio = predicted_volume / effective_capacity if effective_capacity > 0 else 1

    # Determine traffic condition
    if congestion_ratio > 0.8:
        return "🚨 Severe Congestion", congestion_ratio, effective_capacity
    elif congestion_ratio > 0.6:
        return "⚠️ Heavy Traffic", congestion_ratio, effective_capacity
    elif congestion_ratio > 0.4:
        return "🟠 Moderate Traffic", congestion_ratio, effective_capacity
    elif congestion_ratio > 0.2:
        return "🟡 Light Traffic", congestion_ratio, effective_capacity
    else:
        return "🟢 Free Flow", congestion_ratio, effective_capacity


def calculate_overall_risk(traffic_condition, weather, visibility, volume, congestion_ratio):
    """
    Calculate integrated risk level considering all factors
    """
    risk_score = 0

    # Weather risk (0-3 points)
    weather_risk = {
        "Sky is Clear": 0,
        "LightRain": 1,
        "HeavyRain": 3,
        "Fog": 3,
        "Snow": 3,
        "Thunderstorm": 3
    }

    # Traffic condition risk (0-3 points)
    traffic_risk = {
        "🟢 Free Flow": 0,
        "🟡 Light Traffic": 1,
        "🟠 Moderate Traffic": 2,
        "⚠️ Heavy Traffic": 3,
        "🚨 Severe Congestion": 3
    }

    # Visibility risk (0-2 points)
    if visibility < 1:
        visibility_risk = 2
    elif visibility < 3:
        visibility_risk = 1
    else:
        visibility_risk = 0

    # Volume risk adjustment (even low volume in bad conditions is risky)
    if volume < 500 and weather in ["HeavyRain", "Fog", "Snow", "Thunderstorm"]:
        volume_risk = 2  # Low volume but dangerous conditions
    elif volume > 3000:
        volume_risk = 2  # High volume risk
    else:
        volume_risk = 0

    total_risk = (weather_risk.get(weather, 1) +
                  traffic_risk.get(traffic_condition, 1) +
                  visibility_risk +
                  volume_risk)

    # Determine risk level
    if total_risk >= 6:
        return "HIGH"
    elif total_risk >= 3:
        return "MEDIUM"
    else:
        return "LOW"


def get_police_recommendation(traffic_condition, weather_description, congestion_ratio, predicted_volume, visibility,
                              overall_risk):
    """
    Provide specific police deployment recommendations considering integrated risk
    """
    recommendations = []

    # Base recommendations on overall risk level
    if overall_risk == "HIGH":
        recommendations.append("## 🚨 HIGH RISK SITUATION")

        if predicted_volume < 1000 and weather_description in ["HeavyRain", "Fog", "Snow", "Thunderstorm"]:
            recommendations.extend([
                "🚓 **DEPLOYMENT**: Strategic patrols for accident prevention",
                "⚠️ **CONDITIONS**: Low traffic but EXTREMELY hazardous weather",
                "🎯 **FOCUS**: Monitor high-risk areas and accident hotspots",
                "🚧 **ACTIONS**: Deploy warning signs, reduce speed limits if possible",
                "📞 **COORDINATION**: Alert emergency services for quick response",
                "👀 **MONITORING**: Continuous surveillance of hazardous locations"
            ])
        else:
            recommendations.extend([
                "🚓 **DEPLOYMENT**: Maximum traffic police units required",
                "📱 **PLAN**: Activate emergency traffic management protocol",
                "🔄 **ROUTES**: Prepare traffic diversion plans",
                "⏱️ **RESPONSE**: Extended emergency response times expected",
                "🚨 **ALERT**: High probability of incidents and delays",
                "📞 **COORDINATION**: Full emergency services coordination"
            ])

        # Weather-specific high-risk actions
        if weather_description in ["HeavyRain", "Thunderstorm"]:
            recommendations.append("💨 **HAZARD**: Monitor for hydroplaning and flooding areas")
        elif weather_description in ["Fog"]:
            recommendations.append("🌫️ **HAZARD**: Extreme visibility reduction - deploy fog warnings")
        elif weather_description in ["Snow"]:
            recommendations.append("❄️ **HAZARD**: Icy conditions - salt trucks and plows needed")

    elif overall_risk == "MEDIUM":
        recommendations.append("## 🟠 MEDIUM RISK SITUATION")

        recommendations.extend([
            "👮 **DEPLOYMENT**: Additional officers at key intersections",
            "📊 **MONITORING**: Close watch on accident-prone areas",
            "🚦 **CONTROL**: Adjust traffic signals if congestion builds",
            "📢 **COMMS**: Issue traffic and weather advisories to public"
        ])

        if weather_description in ["LightRain", "Mist"]:
            recommendations.append("🌧️ **CAUTION**: Increased patrols on slippery road sections")
        if congestion_ratio > 0.5:
            recommendations.append("🚗 **TRAFFIC**: Building congestion - prepare escalation plan")

    else:  # LOW risk
        recommendations.append("## 🟢 LOW RISK SITUATION")

        recommendations.extend([
            "✅ **DEPLOYMENT**: Normal patrol patterns sufficient",
            "📋 **PROCEDURES**: Standard monitoring protocols",
            "🔍 **VIGILANCE**: Maintain routine surveillance"
        ])

        # Even in low risk, maintain weather awareness
        if weather_description in ["LightRain", "Clouds"]:
            recommendations.append("🌦️ **AWARENESS**: Minor weather conditions - stay alert")

    # Special case: Very specific dangerous scenarios
    if predicted_volume < 500 and weather_description in ["HeavyRain", "Fog", "Snow"] and visibility < 2:
        recommendations.append("\n🚨 **SPECIAL SCENARIO**: EXTREMELY DANGEROUS - Low traffic but lethal conditions")
        recommendations.append("• Focus entirely on accident prevention")
        recommendations.append("• Maximum visibility patrols (lights, signs)")
        recommendations.append("• Prepare for multi-vehicle incidents")

    return recommendations


if st.button("🚗 Analyze Traffic Conditions"):
    # Get volume prediction
    predicted_volume = np.expm1(model.predict(input_df)[0])

    # Calculate traffic condition
    traffic_condition, congestion_ratio, effective_capacity = calculate_traffic_condition(
        predicted_volume, weather_description, visibility, wind_speed
    )

    # Calculate overall integrated risk
    overall_risk = calculate_overall_risk(
        traffic_condition, weather_description, visibility, predicted_volume, congestion_ratio
    )

    # Display results
    st.header("📊 Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Predicted Vehicle Volume", f"{predicted_volume:.0f} vehicles/hr")

    with col2:
        st.metric("Road Capacity in Conditions", f"{effective_capacity:.0f} vehicles/hr")

    with col3:
        st.metric("Congestion Level", f"{congestion_ratio:.1%}")

    with col4:
        risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
        st.metric("Overall Risk Level", overall_risk)

    # Traffic condition alert
    st.header("🚦 Traffic Status")
    if overall_risk == "HIGH":
        st.error(f"# {traffic_condition}")
    elif overall_risk == "MEDIUM":
        st.warning(f"# {traffic_condition}")
    else:
        st.success(f"# {traffic_condition}")

    # Police recommendations
    st.header("👮 Police Deployment Strategy")
    recommendations = get_police_recommendation(
        traffic_condition, weather_description, congestion_ratio,
        predicted_volume, visibility, overall_risk
    )

    for rec in recommendations:
        if rec.startswith("##"):
            st.markdown(rec)
        elif rec.startswith("🚨") or rec.startswith("🚓"):
            st.error(rec)
        elif rec.startswith("👮") or rec.startswith("🟠"):
            st.warning(rec)
        elif rec.startswith("✅") or rec.startswith("🟢"):
            st.success(rec)
        else:
            st.write(rec)

    # Additional insights
    with st.expander("📈 Detailed Analysis Breakdown"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weather Impact")
            weather_reduction = int((1 - (effective_capacity / 5000)) * 100)
            st.write(f"- **Weather**: {weather_description} reduces capacity by {weather_reduction}%")
            st.write(f"- **Visibility**: {visibility} miles - {'Poor' if visibility < 3 else 'Good'} conditions")
            st.write(f"- **Wind**: {wind_speed} mph - {'High' if wind_speed > 20 else 'Normal'} winds")

        with col2:
            st.subheader("Temporal Factors")
            st.write(f"- **Time**: {hour}:00 {'🚗 Rush Hour' if rush_hour else '⏰ Off-Peak'}")
            st.write(f"- **Day**: {'🎉 Weekend' if is_weekend else '📅 Weekday'}")
            st.write(f"- **Holiday**: {holiday_found if holiday_found else 'No holiday'}")

        st.subheader("Risk Factors")
        risk_factors = []
        if weather_description in ["HeavyRain", "Fog", "Snow", "Thunderstorm"]:
            risk_factors.append("🌧️ Hazardous weather conditions")
        if visibility < 3:
            risk_factors.append("👁️ Poor visibility")
        if congestion_ratio > 0.6:
            risk_factors.append("🚗 High traffic congestion")
        if rush_hour:
            risk_factors.append("⏰ Rush hour timing")
        if is_weekend:
            risk_factors.append("🎉 Weekend traffic patterns")

        if risk_factors:
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("- ✅ No significant risk factors identified")

# -------------------------------
# Quick Scenario Examples
# -------------------------------
# st.header("🎯 Common Scenarios")
#
# scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
#
# with scenario_col1:
#     if st.button("🌧️ Rainy Rush Hour"):
#         st.session_state.weather_description = "HeavyRain"
#         st.session_state.hour = 8
#         st.rerun()
#
# with scenario_col2:
#     if st.button("❄️ Snowy Evening"):
#         st.session_state.weather_description = "Snow"
#         st.session_state.hour = 17
#         st.rerun()
#
# with scenario_col3:
#     if st.button("🌫️ Foggy Morning"):
#         st.session_state.weather_description = "Fog"
#         st.session_state.visibility = 1
#         st.session_state.hour = 7
#         st.rerun()

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🚦 Smart Traffic Prediction System | For Police Deployment Optimization
    </div>
    """,
    unsafe_allow_html=True
)