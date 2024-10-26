import pandas as pd
import joblib

# Load the saved model and scaler
rf_model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new data
new_data = pd.read_csv('new_data.csv')  # Replace 'new_data.csv' with your new dataset

# Select relevant columns and ensure they match the training data
columns_to_use = [
    'HS Op', 'HS Dr', 'HS Goal', 'TC DB2', 'TC SF1Pressure',
    'TC SF1Upper Corr', 'TC SF1Lower Corr', 'TC SF1Preheater',
    'TC SF2Pressure', 'TC SF2Upper Corr', 'TC SF2Lower Corr', 'TC SF2Preheater'
]

# Extract the selected columns and drop any rows with missing values
new_data = new_data[columns_to_use].dropna()

# Calculate deviations between actual and goal values
new_data['HS_Deviation'] = new_data['HS Dr'] - new_data['HS Goal']
new_data['SF1_Pressure_Deviation'] = new_data['TC SF1Pressure'] - new_data['TC SF2Pressure']

# Standardize the features using the previously saved scaler
new_data_scaled = scaler.transform(new_data)

# Predict using the loaded Random Forest model
predictions = rf_model.predict(new_data_scaled)

# Output the predictions
print("Predicted Speeds:")
print(predictions)
