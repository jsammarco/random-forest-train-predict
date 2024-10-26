import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('multi_output_rf_model_beds_baths.pkl')
scaler = joblib.load('scaler_beds_baths.pkl')

# Load new data
new_data = pd.read_csv("new_data.csv")

# Clean and process Price column
new_data['Price'] = pd.to_numeric(new_data['Price'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
new_data['Price'] = new_data['Price'] / 1000

# Select only the relevant columns for prediction
X_new = new_data[['Price', 'Sqft']]

# Scale the new data
X_new_scaled = scaler.transform(X_new)

# Make predictions and round them
predictions = model.predict(X_new_scaled)
rounded_predictions = np.round(predictions)

# Calculate prediction confidence as the standard deviation from each tree for each target
confidence_intervals = []
for x in X_new_scaled:
    # Collect predictions from each regressor in MultiOutputRegressor
    beds_predictions = [est.predict([x])[0] for est in model.estimators_[0].estimators_]  # predictions for 'Beds'
    baths_predictions = [est.predict([x])[0] for est in model.estimators_[1].estimators_]  # predictions for 'Baths'
    
    # Calculate standard deviation for each target as a proxy for confidence
    beds_std = np.std(beds_predictions)
    baths_std = np.std(baths_predictions)
    confidence_intervals.append([beds_std, baths_std])

# Convert predictions and confidence intervals to DataFrames
predicted_df = pd.DataFrame(rounded_predictions, columns=['Predicted_Beds', 'Predicted_Baths']).reset_index(drop=True)
confidence_df = pd.DataFrame(confidence_intervals, columns=['Confidence_Beds', 'Confidence_Baths']).reset_index(drop=True)

# Combine with original data and display
result = pd.concat([new_data[['Price', 'Sqft']].reset_index(drop=True), predicted_df, confidence_df], axis=1)

print("Predicted Beds and Baths with Confidence based on Price and Sqft:")
print(result)
