import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
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
# OpenWeather API Configuration
# -------------------------------
st.sidebar.header("⚙️ API Configuration")
api_key = st.sidebar.text_input("OpenWeather API Key", type="password",
                                help="Get your free API key from openweathermap.org")
city = st.sidebar.text_input("City", "Minneapolis", help="Enter city name for weather data")
use_live_weather = st.sidebar.checkbox("Use Live Weather Data", value=False)


def fetch_weather_data(api_key, city, selected_datetime):
    """
    Fetch weather data from OpenWeather API
    For current/future times: uses current weather + forecast
    For past times: uses current weather as approximation
    """
    try:
        # Current weather endpoint
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=imperial"

        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract weather data
        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'visibility': data['visibility'] / 1609.34,  # Convert meters to miles
            'wind_speed': data['wind']['speed'],
            'weather_type': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description']
        }

        # Check if we should use forecast data for future predictions
        time_diff = (selected_datetime - datetime.now()).total_seconds() / 3600  # hours

        if 0 < time_diff <= 120:  # Up to 5 days in future
            try:
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=imperial"
                forecast_response = requests.get(forecast_url, timeout=10)
                forecast_response.raise_for_status()
                forecast_data = forecast_response.json()

                # Find closest forecast to selected time
                target_timestamp = selected_datetime.timestamp()
                closest_forecast = min(
                    forecast_data['list'],
                    key=lambda x: abs(x['dt'] - target_timestamp)
                )

                # Update with forecast data
                weather_data.update({
                    'temperature': closest_forecast['main']['temp'],
                    'humidity': closest_forecast['main']['humidity'],
                    'visibility': closest_forecast.get('visibility', 10000) / 1609.34,
                    'wind_speed': closest_forecast['wind']['speed'],
                    'weather_type': closest_forecast['weather'][0]['main'],
                    'weather_description': closest_forecast['weather'][0]['description']
                })

                st.sidebar.success(f"✅ Using forecast data for {selected_datetime.strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                st.sidebar.info("📊 Using current weather data (forecast unavailable)")

        elif time_diff <= 0:
            st.sidebar.info("📊 Using current weather data (historical data not available)")
        else:
            st.sidebar.warning("⚠️ Selected time too far in future. Using current weather.")

        return weather_data, None

    except requests.exceptions.RequestException as e:
        return None, f"API Error: {str(e)}"
    except KeyError as e:
        return None, f"Data parsing error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def map_weather_description(raw_description):
    """Map OpenWeather descriptions to model categories"""
    desc_lower = raw_description.lower()

    if 'clear' in desc_lower:
        return 'Sky is Clear'
    elif 'rain' in desc_lower:
        if 'heavy' in desc_lower or 'extreme' in desc_lower:
            return 'HeavyRain'
        else:
            return 'LightRain'
    elif 'fog' in desc_lower or 'mist' in desc_lower:
        return 'Fog'
    elif 'snow' in desc_lower:
        return 'Snow'
    elif 'thunder' in desc_lower:
        return 'Thunderstorm'
    else:
        return 'Sky is Clear'


def map_weather_type(raw_type):
    """Map OpenWeather types to model categories"""
    type_mapping = {
        'Clear': 'Clear',
        'Clouds': 'Clouds',
        'Rain': 'Rain',
        'Drizzle': 'Rain',
        'Snow': 'Snow',
        'Mist': 'Mist',
        'Fog': 'Fog',
        'Haze': 'Mist',
        'Thunderstorm': 'Thunderstorm',
        'Smoke': 'Mist'
    }
    return type_mapping.get(raw_type, 'Clear')


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
day_of_week = selected_datetime.weekday()
month = selected_datetime.month
year = selected_datetime.year
is_weekend = 1 if day_of_week >= 5 else 0

# -------------------------------
# Weather Inputs (Manual or API)
# -------------------------------
st.header("🌦 Weather Conditions")

# Initialize weather variables
humidity = 70
wind_speed = 5
visibility = 10
temperature = 75
weather_type = "Clear"
weather_description = "Sky is Clear"

if use_live_weather and api_key:
    with st.spinner("🌐 Fetching live weather data..."):
        weather_data, error = fetch_weather_data(api_key, city, selected_datetime)

        if error:
            st.error(f"❌ {error}")
            st.info("📝 Please enter weather data manually below")
        else:
            st.success(f"✅ Live weather data loaded for {city}")

            # Update variables with API data
            temperature = weather_data['temperature']
            humidity = weather_data['humidity']
            visibility = min(weather_data['visibility'], 20)  # Cap at 20 miles
            wind_speed = min(weather_data['wind_speed'], 30)  # Cap at 30 mph
            weather_type = map_weather_type(weather_data['weather_type'])
            weather_description = map_weather_description(weather_data['weather_description'])

            # Display fetched data
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌡️ Temperature", f"{temperature:.1f}°F")
            with col2:
                st.metric("💧 Humidity", f"{humidity}%")
            with col3:
                st.metric("👁️ Visibility", f"{visibility:.1f} mi")
            with col4:
                st.metric("💨 Wind Speed", f"{wind_speed:.1f} mph")

            st.info(f"☁️ Weather: {weather_type} - {weather_data['weather_description']}")

# Manual input option (always shown for override)
with st.expander("🔧 Manual Weather Override" if use_live_weather and api_key else "📝 Enter Weather Data Manually"):
    col1, col2, col3 = st.columns(3)
    with col1:
        humidity = st.slider("Humidity (%)", 0, 100, int(humidity))
        wind_speed = st.slider("Wind Speed (mph)", 0, 30, int(wind_speed))
    with col2:
        visibility = st.slider("Visibility (miles)", 0, 20, int(visibility))
        temperature = st.slider("Temperature (°F)", -10, 120, int(temperature))
    with col3:
        weather_type = st.selectbox(
            "Weather Type",
            ["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist", "Thunderstorm"],
            index=["Clear", "Clouds", "Rain", "Snow", "Fog", "Mist", "Thunderstorm"].index(weather_type)
        )
        weather_description = st.selectbox(
            "Weather Description",
            ["Sky is Clear", "LightRain", "HeavyRain", "Fog", "Snow", "Thunderstorm"],
            index=["Sky is Clear", "LightRain", "HeavyRain", "Fog", "Snow", "Thunderstorm"].index(weather_description)
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

input_data.update(is_holiday_cols)

input_df = pd.DataFrame([input_data])
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[training_columns]


# -------------------------------
# Traffic Analysis Functions
# -------------------------------
def calculate_traffic_condition(predicted_volume, weather_condition, visibility, wind_speed):
    base_capacity = 5000
    weather_impact = {
        "Sky is Clear": 1.0,
        "LightRain": 0.7,
        "HeavyRain": 0.5,
        "Fog": 0.4,
        "Snow": 0.3,
        "Thunderstorm": 0.4
    }

    if visibility < 1:
        visibility_factor = 0.4
    elif visibility < 3:
        visibility_factor = 0.6
    elif visibility < 5:
        visibility_factor = 0.8
    else:
        visibility_factor = 1.0

    wind_factor = 0.8 if wind_speed > 20 else 1.0
    weather_factor = weather_impact.get(weather_condition, 0.7)
    effective_capacity = base_capacity * weather_factor * visibility_factor * wind_factor
    congestion_ratio = predicted_volume / effective_capacity if effective_capacity > 0 else 1

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
    risk_score = 0
    weather_risk = {
        "Sky is Clear": 0,
        "LightRain": 1,
        "HeavyRain": 3,
        "Fog": 3,
        "Snow": 3,
        "Thunderstorm": 3
    }
    traffic_risk = {
        "🟢 Free Flow": 0,
        "🟡 Light Traffic": 1,
        "🟠 Moderate Traffic": 2,
        "⚠️ Heavy Traffic": 3,
        "🚨 Severe Congestion": 3
    }

    visibility_risk = 2 if visibility < 1 else (1 if visibility < 3 else 0)
    volume_risk = 2 if (volume < 500 and weather in ["HeavyRain", "Fog", "Snow",
                                                     "Thunderstorm"]) or volume > 3000 else 0

    total_risk = (weather_risk.get(weather, 1) + traffic_risk.get(traffic_condition, 1) +
                  visibility_risk + volume_risk)

    return "HIGH" if total_risk >= 6 else ("MEDIUM" if total_risk >= 3 else "LOW")


def get_police_recommendation(traffic_condition, weather_description, congestion_ratio,
                              predicted_volume, visibility, overall_risk):
    recommendations = []

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

    else:
        recommendations.append("## 🟢 LOW RISK SITUATION")
        recommendations.extend([
            "✅ **DEPLOYMENT**: Normal patrol patterns sufficient",
            "📋 **PROCEDURES**: Standard monitoring protocols",
            "🔍 **VIGILANCE**: Maintain routine surveillance"
        ])

        if weather_description in ["LightRain", "Clouds"]:
            recommendations.append("🌦️ **AWARENESS**: Minor weather conditions - stay alert")

    if predicted_volume < 500 and weather_description in ["HeavyRain", "Fog", "Snow"] and visibility < 2:
        recommendations.append("\n🚨 **SPECIAL SCENARIO**: EXTREMELY DANGEROUS - Low traffic but lethal conditions")
        recommendations.append("• Focus entirely on accident prevention")
        recommendations.append("• Maximum visibility patrols (lights, signs)")
        recommendations.append("• Prepare for multi-vehicle incidents")

    return recommendations


# -------------------------------
# Analysis Button
# -------------------------------
st.header("🔮 Traffic Condition Analysis")

if st.button("🚗 Analyze Traffic Conditions"):
    predicted_volume = np.expm1(model.predict(input_df)[0])

    traffic_condition, congestion_ratio, effective_capacity = calculate_traffic_condition(
        predicted_volume, weather_description, visibility, wind_speed
    )

    overall_risk = calculate_overall_risk(
        traffic_condition, weather_description, visibility, predicted_volume, congestion_ratio
    )

    st.header("📊 Prediction Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Predicted Vehicle Volume", f"{predicted_volume:.0f} vehicles/hr")
    with col2:
        st.metric("Road Capacity in Conditions", f"{effective_capacity:.0f} vehicles/hr")
    with col3:
        st.metric("Congestion Level", f"{congestion_ratio:.1%}")
    with col4:
        st.metric("Overall Risk Level", overall_risk)

    st.header("🚦 Traffic Status")
    if overall_risk == "HIGH":
        st.error(f"# {traffic_condition}")
    elif overall_risk == "MEDIUM":
        st.warning(f"# {traffic_condition}")
    else:
        st.success(f"# {traffic_condition}")

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

    with st.expander("📈 Detailed Analysis Breakdown"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Weather Impact")
            weather_reduction = int((1 - (effective_capacity / 5000)) * 100)
            st.write(f"- **Weather**: {weather_description} reduces capacity by {weather_reduction}%")
            st.write(f"- **Visibility**: {visibility:.1f} miles - {'Poor' if visibility < 3 else 'Good'} conditions")
            st.write(f"- **Wind**: {wind_speed:.1f} mph - {'High' if wind_speed > 20 else 'Normal'} winds")

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

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🚦 Smart Traffic Prediction System | For Police Deployment Optimization
    </div>
    """,
    unsafe_allow_html=True
)