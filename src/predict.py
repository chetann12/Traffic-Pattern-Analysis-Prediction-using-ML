# src/predict.py
import pandas as pd
import joblib
import os
import numpy as np

def make_prediction(input_data_df, model_name='random_forest_model', models_dir="../models/"):
    """
    Loads a specified trained model and makes predictions on new input data.

    Args:
        input_data_df (pd.DataFrame): DataFrame containing the features for prediction.
                                      Must have the same columns as the training data,
                                      excluding 'traffic_volume' and 'date_time'.
        model_name (str): The name of the model to use (e.g., 'random_forest_model',
                          'xgboost_model', 'lightgbm_model').
        models_dir (str): Directory where models are saved.

    Returns:
        np.array: Predicted traffic volume.
    """
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    training_columns_path = os.path.join(models_dir, 'training_columns.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_name}.pkl' not found at {model_path}.")
    if not os.path.exists(training_columns_path):
        raise FileNotFoundError(f"Training columns file not found at {training_columns_path}. Please run 'train_model.py' first.")

    print(f"Loading {model_name.replace('_', ' ').title()} for prediction...")
    model = joblib.load(model_path)
    training_columns = joblib.load(training_columns_path)

    # Align input data to training features
    for col in training_columns:
        if col not in input_data_df.columns:
            input_data_df[col] = 0
    input_data_df = input_data_df[training_columns]

    # Ensure numeric types
    input_data_df = input_data_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Make predictions
    predictions_log = model.predict(input_data_df)
    predictions = np.expm1(predictions_log)  # inverse log1p

    # Cap extremely large values to avoid overflow
    predictions = np.clip(predictions, 0, 1e6)

    return predictions

if __name__ == "__main__":
    # Example usage
    print("--- Example Prediction ---")
    try:
        X_test_sample_path = os.path.join("../models/", 'X_test.csv')
        if not os.path.exists(X_test_sample_path):
            print(f"'{X_test_sample_path}' not found. Please run 'train_model.py' first.")
            exit()

        X_sample = pd.read_csv(X_test_sample_path).sample(5, random_state=42)  # 5 random samples
        print("\nSample input data for prediction (first 5 rows from test set):")
        print(X_sample.head())

        # Predict using all models
        for model in ['random_forest_model', 'xgboost_model', 'lightgbm_model']:
            preds = make_prediction(X_sample, model_name=model)
            print(f"\n{model.replace('_', ' ').title()} predictions for sample:\n{preds}")



    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
