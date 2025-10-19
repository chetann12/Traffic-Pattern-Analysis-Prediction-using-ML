import pandas as pd
import joblib

# # Example: load your dataset
# df = pd.read_csv('data/clean.csv')
#
# # Print all column names
# print("Total columns:", len(df.columns))
# print(df.columns.tolist())



cols = joblib.load('../models/training_columns.pkl')
print("Total columns:", len(cols))
print(cols)
