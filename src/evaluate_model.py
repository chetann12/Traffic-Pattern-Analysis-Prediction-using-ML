# src/evaluate_model.py
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

def evaluate_models(models_dir="../models/"):
    """
    Loads trained models and test data, aligns features, makes predictions,
    and evaluates performance. Handles log-transformed targets.
    """
    # Load test data
    X_test = pd.read_csv(os.path.join(models_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(models_dir, 'y_test.csv')).squeeze()  # Series

    # Load training feature columns for alignment
    training_columns_path = os.path.join(models_dir, 'training_columns.pkl')
    if not os.path.exists(training_columns_path):
        raise FileNotFoundError(f"Training columns file not found at {training_columns_path}. Please run 'train_model.py' first.")

    training_columns = joblib.load(training_columns_path)

    # Align test data to training features
    # Add missing columns with 0 and order columns as in training
    for col in training_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[training_columns]

    # Ensure numeric types
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Since we used log1p during training, apply inverse log for metrics
    y_test_exp = np.expm1(y_test)

    model_names = ['random_forest_model', 'xgboost_model', 'lightgbm_model']
    results = {}

    for model_name in model_names:
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        if not os.path.exists(model_path):
            print(f"Model {model_name}.pkl not found. Skipping evaluation.")
            continue

        print(f"\nEvaluating {model_name.replace('_', ' ').title()}...")
        model = joblib.load(model_path)

        # Predict and inverse log1p
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        # Cap extremely large values to avoid overflow issues
        y_pred = np.clip(y_pred, 0, 1e6)

        # Metrics
        r2 = r2_score(y_test_exp, y_pred)
        mae = mean_absolute_error(y_test_exp, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred))

        results[model_name] = {'R-squared': r2, 'MAE': mae, 'RMSE': rmse}

        print(f"  R-squared: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")

    # Model comparison
    print("\n--- Model Comparison ---")
    results_df = pd.DataFrame.from_dict(results, orient='index')
    print(results_df)

    # Best models
    if not results_df.empty:
        best_r2 = results_df['R-squared'].idxmax()
        best_mae = results_df['MAE'].idxmin()
        best_rmse = results_df['RMSE'].idxmin()

        print(f"\nBest model by R-squared: {best_r2.replace('_', ' ').title()} with R² = {results_df.loc[best_r2, 'R-squared']:.4f}")
        print(f"Best model by MAE: {best_mae.replace('_', ' ').title()} with MAE = {results_df.loc[best_mae, 'MAE']:.2f}")
        print(f"Best model by RMSE: {best_rmse.replace('_', ' ').title()} with RMSE = {results_df.loc[best_rmse, 'RMSE']:.2f}")
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    evaluate_models()
