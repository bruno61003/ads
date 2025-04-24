import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("linear_advertising_data.csv")  # Replace with actual file path

# Splitting features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

# MASE Calculation
y_naive_pred = np.roll(y_train, shift=1)  # Naïve forecast (shifted previous values)
mase_denom = np.mean(np.abs(y_train[1:] - y_naive_pred[1:]))  # Scale factor
mase = mae / mase_denom if mase_denom != 0 else np.nan  # Avoid division by zero

# Print results
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("MAE in %:", (mae / np.mean(y_test)) * 100)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Mean Absolute Scaled Error (MASE):", mase)
print("R-squared Score (R²):", r2_score(y_test, y_pred))

# Scatter plots for linearity check
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(X.columns):
    axes[idx].scatter(df[feature], y, color='blue', alpha=0.6)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Sales')
    axes[idx].set_title(f'Sales vs {feature}')

plt.tight_layout()
plt.show()