import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model, scaler, and feature columns
multi_rf_model = joblib.load('multi_output_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load the new data
new_data_file = 'new_data.csv'
new_data = pd.read_csv(new_data_file)

# Preprocess the new data using the same steps as during training
# Drop the same columns as in the training script
columns_to_drop = ['Date', 'Grade', 'DB Grade', 'Id Code', 'Flute', 'SF1 Liner', 'SF1 Medium',
                   'SF1 Flute', 'LastModified']
new_data = new_data.drop(columns=columns_to_drop, errors='ignore')

# Remove "%" from the percentage columns and convert to numeric
percent_columns = [col for col in new_data.columns if 'Percent' in col]
for col in percent_columns:
    if new_data[col].dtype == 'object':
        new_data[col] = new_data[col].str.replace('%', '').astype(float)

# Remove "Auto/Manual/On/Off" from the state columns and convert to numeric
auto_columns = [col for col in new_data.columns if 'State' in col]
state_mappings = {'Auto': 1, 'On': 1, 'Manual': 0, 'Off': 0}
for col in auto_columns:
    if new_data[col].dtype == 'object':
        new_data[col] = new_data[col].replace(state_mappings).astype(float)

# Convert 'HS Dr' and 'HS Goal' to numeric, coercing any errors
new_data['HS Dr'] = pd.to_numeric(new_data['HS Dr'], errors='coerce')
new_data['HS Goal'] = pd.to_numeric(new_data['HS Goal'], errors='coerce')

# Calculate the deviation
new_data['HS_Deviation'] = new_data['HS Dr'] - new_data['HS Goal']

# Target columns (these are what the model predicts)
target_columns = ['HS Goal', 'TC SF1Preheater', 'TC SF2Preheater']

# Features (exclude target columns if they exist in new_data)
X_new = new_data.drop(columns=target_columns, errors='ignore')

# Identify object-type columns in X_new
object_columns = X_new.select_dtypes(include=['object']).columns.tolist()

# Process object-type columns
for col in object_columns:
    # Remove '%' if present
    if X_new[col].str.contains('%').any():
        X_new[col] = X_new[col].str.replace('%', '')
    # Map 'Auto', 'Manual', 'On', 'Off' to numeric
    X_new[col] = X_new[col].replace(state_mappings)
    # Attempt to convert to numeric
    X_new[col] = pd.to_numeric(X_new[col], errors='coerce')
    # Handle any remaining non-numeric values
    if X_new[col].isna().any():
        X_new[col].fillna(X_new[col].mean(), inplace=True)

# After processing, check for any remaining object-type columns
object_columns = X_new.select_dtypes(include=['object']).columns.tolist()
if object_columns:
    X_new.drop(columns=object_columns, inplace=True)

# Align features with those used during training
# Reindex the DataFrame to have the same columns as in the training data
X_new = X_new.reindex(columns=feature_columns)

# Handle any new columns or missing columns
missing_cols = X_new.columns[X_new.isna().any()].tolist()
if missing_cols:
    # Fill missing values with zero or an appropriate value
    X_new[missing_cols] = X_new[missing_cols].fillna(0)

# Standardize the features using the loaded scaler
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = multi_rf_model.predict(X_new_scaled)

# Create a DataFrame to hold the predictions
predictions_df = pd.DataFrame(predictions, columns=target_columns)

# Include any identifiers from the new data, such as 'RunNumber'
if 'RunNumber' in new_data.columns:
    predictions_df.insert(0, 'RunNumber', new_data['RunNumber'].values)

# Output the predictions
print(predictions_df)

# Optionally, save the predictions to a CSV file
predictions_df.to_csv('predictions.csv', index=False)
