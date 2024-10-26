import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
file_path = "9760.csv"
data = pd.read_csv(file_path)

# Select relevant columns
columns_to_use = [
    'Speed', 'HS Op', 'HS Dr', 'HS Goal', 'TC DB2', 'TC SF1Pressure',
    'TC SF1Upper Corr', 'TC SF1Lower Corr', 'TC SF1Preheater',
    'TC SF2Pressure', 'TC SF2Upper Corr', 'TC SF2Lower Corr', 'TC SF2Preheater'
]

# Extract the selected columns
data = data[columns_to_use].dropna()

# Calculate deviations between actual and goal values
data['HS_Deviation'] = data['HS Dr'] - data['HS Goal']
data['SF1_Pressure_Deviation'] = data['TC SF1Pressure'] - data['TC SF2Pressure']

# Define features (X) and target (y)
X = data.drop(['Speed'], axis=1)
y = data['Speed']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest Mean Squared Error: {mse_rf}")
print(f"Random Forest R² Score: {r2_rf}")
