# data_preprocessing_and_split.py
import pandas as pd
import numpy as np

def load_and_clean_data(path, cap_traffic=True):
    df = pd.read_csv(path)

    # -------------------------
    # Handle missing values
    # -------------------------
    df['is_holiday'] = df['is_holiday'].fillna("None")
    df = df.fillna(df.median(numeric_only=True))

    # -------------------------
    # Convert date_time
    # -------------------------
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['rush_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 16 <= x <= 19 else 0)

    # -------------------------
    # Consolidate 'weather_description'
    # -------------------------
    weather_mapping = {
        'sky is clear': 'Clear', 'few clouds': 'Clear', 'scattered clouds': 'Clear', 'broken clouds': 'Clear',
        'overcast clouds': 'Clear', 'haze': 'Clear', 'smoke': 'Clear', 'mist': 'Clear',
        'drizzle': 'LightRain', 'light intensity drizzle': 'LightRain', 'light rain': 'LightRain',
        'light intensity shower rain': 'LightRain', 'shower drizzle': 'LightRain', 'proximity shower rain': 'LightRain',
        'moderate rain': 'HeavyRain', 'heavy intensity drizzle': 'HeavyRain', 'heavy intensity rain': 'HeavyRain',
        'very heavy rain': 'HeavyRain',
        'light snow': 'Snow', 'snow': 'Snow', 'heavy snow': 'Snow', 'sleet': 'Snow', 'freezing rain': 'Snow',
        'light rain and snow': 'Snow', 'light shower snow': 'Snow', 'shower snow': 'Snow',
        'thunderstorm': 'Thunderstorm', 'proximity thunderstorm': 'Thunderstorm',
        'thunderstorm with drizzle': 'Thunderstorm',
        'thunderstorm with light drizzle': 'Thunderstorm', 'thunderstorm with rain': 'Thunderstorm',
        'thunderstorm with light rain': 'Thunderstorm', 'thunderstorm with heavy rain': 'Thunderstorm',
        'proximity thunderstorm with drizzle': 'Thunderstorm', 'proximity thunderstorm with rain': 'Thunderstorm',
        'fog': 'Fog', 'SQUALLS': 'Squalls'
    }
    df['weather_description'] = df['weather_description'].replace(weather_mapping)
    df = pd.get_dummies(df, columns=['weather_description'], drop_first=True)

    # -------------------------
    # Encode weather_type
    # -------------------------
    if 'weather_type' in df.columns:
        df = pd.get_dummies(df, columns=['weather_type'], drop_first=True)

    # -------------------------
    # Encode is_holiday
    # -------------------------
    df = pd.get_dummies(df, columns=['is_holiday'], drop_first=True)

    # -------------------------
    # Drop unwanted columns
    # -------------------------
    df = df.drop(columns=['air_pollution_index', 'wind_direction', 'dew_point', 'rain_p_h', 'snow_p_h', 'clouds_all'], errors='ignore')

    # -------------------------
    # Interaction feature example
    # -------------------------
    df['temp_humidity'] = df['temperature'] * df['humidity']

    # -------------------------
    # Optional: cap traffic_volume to reduce extreme outliers
    # -------------------------
    if cap_traffic and 'traffic_volume' in df.columns:
        cap_value = df['traffic_volume'].quantile(0.99)
        df['traffic_volume'] = np.minimum(df['traffic_volume'], cap_value)

    return df


def create_lag_features(df):
    df = df.sort_values('date_time').copy()
    df['traffic_prev_hour'] = df['traffic_volume'].shift(1)
    df['traffic_prev_day_same_hour'] = df['traffic_volume'].shift(24)
    df = df.fillna(0)  # For train, first row(s) will have NaN
    return df


def split_train_test(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    # -------------------------
    # Lag features for test using last rows of train
    # -------------------------
    last_train_rows = train_df.tail(24)  # at least 24 rows for 1-day lag
    test_combined = pd.concat([last_train_rows, test_df], ignore_index=True)
    test_combined['traffic_prev_hour'] = test_combined['traffic_volume'].shift(1)
    test_combined['traffic_prev_day_same_hour'] = test_combined['traffic_volume'].shift(24)

    # Extract only actual test rows
    test_df = test_combined.iloc[24:].copy()  # drop extra rows used for lag
    test_df = test_df.fillna(0)

    return train_df, test_df


if __name__ == "__main__":
    path = "../data/raw.csv"
    df = load_and_clean_data(path, cap_traffic=True)
    df = create_lag_features(df)  # For full dataset (train + lag features)

    train_df, test_df = split_train_test(df)

    train_df.to_csv("../data/train.csv", index=False)
    test_df.to_csv("../data/test.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print("Train and test sets saved to ../data/train.csv and ../data/test.csv")
