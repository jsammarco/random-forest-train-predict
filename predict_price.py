import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('rf_model_price.pkl')
scaler = joblib.load('scaler_price.pkl')

# Load new data
new_data = pd.read_csv("new_data.csv")

# Fill missing values for Beds and Baths with 0
new_data['Beds'] = new_data['Beds'].fillna(0)
new_data['Baths'] = new_data['Baths'].fillna(0)

# Select only the relevant columns for prediction
X_new = new_data[['Beds', 'Baths', 'Sqft']]

# Scale the new data
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)
rounded_predictions = np.round(predictions)

# Calculate prediction confidence as the standard deviation of predictions from each tree
confidence_intervals = []
for x in X_new_scaled:
    # Get predictions from each tree for the current sample
    tree_predictions = [est.predict([x])[0] for est in model.estimators_]
    # Calculate standard deviation as a proxy for confidence
    price_std = np.std(tree_predictions)
    confidence_intervals.append(price_std)

# Convert predictions and confidence intervals to DataFrames
predicted_df = pd.DataFrame(rounded_predictions, columns=['Predicted_Price']).reset_index(drop=True)
confidence_df = pd.DataFrame(confidence_intervals, columns=['Confidence_Price']).reset_index(drop=True)

# Combine with original data and display
result = pd.concat([new_data[['Beds', 'Baths', 'Sqft']].reset_index(drop=True), predicted_df, confidence_df], axis=1)

print("Predicted Price with Confidence based on Beds, Baths, and Sqft:")
print(result)
