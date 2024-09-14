import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

# Load environment arguments to get data_path
with open("./madrl/args/env_args/flex_provision.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"]

# Define file names
input_file_name = "net_power_inputs.csv"
output_file_name = "bus_voltages_outputs.csv"

# Construct full file paths
input_file_path = os.path.join(data_path, input_file_name)
output_file_path = os.path.join(data_path, output_file_name)

# Load data
input_df = pd.read_csv(input_file_path)
output_df = pd.read_csv(output_file_path)

# Extract input and output arrays
X = input_df.values
Y = output_df.values

# Normalize data using Min-Max Scaling
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_scaled, test_size=0.2, random_state=42
)

# Define the linear multi-output regressor
linear_regressor = LinearRegression()
model = MultiOutputRegressor(linear_regressor)

# Perform cross-validation on the training data
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mse_scores = cross_val_score(
    model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error'
)
cv_mae_scores = cross_val_score(
    model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error'
)
cv_mape_scores = cross_val_score(
    model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_percentage_error'
)

# Convert scores to positive values
cv_mse_scores = -cv_mse_scores
cv_mae_scores = -cv_mae_scores
cv_mape_scores = -cv_mape_scores

print(f"Cross-validation MSE scores: {cv_mse_scores}")
print(f"Average cross-validation MSE: {np.mean(cv_mse_scores)}")
print(f"Cross-validation MAE scores: {cv_mae_scores}")
print(f"Average cross-validation MAE: {np.mean(cv_mae_scores)}")
print(f"Cross-validation MAPE scores: {cv_mape_scores}")
print(f"Average cross-validation MAPE: {np.mean(cv_mape_scores)}")

# Train the model on the full training set
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Evaluate the model using multiple metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
mape = mean_absolute_percentage_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\nModel Performance on Test Set:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"R² Score: {r2}")

# Save the trained model
model_file_path = "./linear_multioutput_regressor.pkl"
joblib.dump(model, model_file_path)

print(f"\nTrained model saved to: {model_file_path}")