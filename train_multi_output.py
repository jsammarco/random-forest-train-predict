import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data
file_path = "Quoted_60404_Housing_Data.csv"
data = pd.read_csv(file_path, on_bad_lines='skip')
data = data.drop(["Features", "Agent"], axis=1)
# Ensure 'Price' and 'Sqft' columns are numeric, dropping rows with non-numeric data
data['Price'] = pd.to_numeric(data['Price'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
data['Price'] = data['Price']/1000
data['Sqft'] = pd.to_numeric(data['Sqft'], errors='coerce').fillna(0)
data['Beds'] = data['Beds'].fillna(0)
data['Baths'] = data['Baths'].fillna(0)
print(data)
# Drop rows with NaN values in 'Price' and 'Sqft' after conversion
data = data.dropna(subset=['Price', 'Sqft'])

# Now proceed with selecting features and targets as before
X_beds_baths = data[['Price', 'Sqft']]
y_beds_baths = data[['Beds', 'Baths']]
X_price = data[['Beds', 'Baths', 'Sqft']]
y_price = data[['Price']]

# Continue with train-test split and scaling as in previous code


# --- Model 1: Predicting 'beds' and 'baths' with 'price' and 'sqft' ---

# Split data
X_train_bb, X_test_bb, y_train_bb, y_test_bb = train_test_split(X_beds_baths, y_beds_baths, test_size=0.2, random_state=42)

# Standardize the features
scaler_bb = StandardScaler()
X_train_bb_scaled = scaler_bb.fit_transform(X_train_bb)
X_test_bb_scaled = scaler_bb.transform(X_test_bb)

# Train a multi-output Random Forest model
multi_rf_model_bb = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
multi_rf_model_bb.fit(X_train_bb_scaled, y_train_bb)

# Save model and scaler
joblib.dump(multi_rf_model_bb, 'multi_output_rf_model_beds_baths.pkl')
joblib.dump(scaler_bb, 'scaler_beds_baths.pkl')

# Make predictions on the test set
y_pred_bb = multi_rf_model_bb.predict(X_test_bb_scaled)

# Evaluate the model
mse_bb = mean_squared_error(y_test_bb, y_pred_bb)
r2_bb = r2_score(y_test_bb, y_pred_bb)
print(f"Model 1 - Predicting Beds and Baths - Mean Squared Error: {mse_bb}")
print(f"Model 1 - Predicting Beds and Baths - R² Score: {r2_bb}")

# --- Model 2: Predicting 'price' based on 'beds', 'baths', and 'sqft' ---

# Split data
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, random_state=42)

# Standardize the features
scaler_price = StandardScaler()
X_train_price_scaled = scaler_price.fit_transform(X_train_price)
X_test_price_scaled = scaler_price.transform(X_test_price)

# Train the model
rf_model_price = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_price.fit(X_train_price_scaled, y_train_price.values.ravel())

# Save model and scaler
joblib.dump(rf_model_price, 'rf_model_price.pkl')
joblib.dump(scaler_price, 'scaler_price.pkl')

# Make predictions on the test set
y_pred_price = rf_model_price.predict(X_test_price_scaled)

# Evaluate the model
mse_price = mean_squared_error(y_test_price, y_pred_price)
r2_price = r2_score(y_test_price, y_pred_price)
print(f"Model 2 - Predicting Price - Mean Squared Error: {mse_price}")
print(f"Model 2 - Predicting Price - R² Score: {r2_price}")
