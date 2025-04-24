import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = "car_sales.xlsx" # Change to your actual file path
car_sales_df = pd.read_excel(file_path)

# Convert Month column to datetime index
car_sales_df["Month"] = pd.to_datetime(car_sales_df["Month"])
car_sales_df.set_index("Month", inplace=True)

# Plot the original time series data
plt.figure(figsize=(12, 5))
plt.plot(car_sales_df["Sales"], label="Car Sales")
plt.title("Original Car Sales Time Series Data")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# ARIMA Model
arima_model = ARIMA(car_sales_df["Sales"], order=(2, 1, 2)).fit()

# Linear Regression
X = np.arange(len(car_sales_df)).reshape(-1, 1) # Time as the independent
variable
y = car_sales_df["Sales"].values
lin_reg = LinearRegression().fit(X, y)

# Future Forecasting
future_dates = pd.date_range(start=car_sales_df.index[-1] +
pd.DateOffset(months=1), periods=12, freq='M')
future_X = np.arange(len(car_sales_df), len(car_sales_df) + 12).reshape(-1, 1)

# Forecasts
arima_forecast = arima_model.forecast(steps=12)
lin_reg_forecast = lin_reg.predict(future_X)

# Create forecast DataFrame
forecast_df = pd.DataFrame({
"Month": future_dates,
"ARIMA_Forecast": arima_forecast,
"Linear_Regression_Forecast": lin_reg_forecast
})

# Plot Forecasts
plt.figure(figsize=(12, 5))
plt.plot(car_sales_df["Sales"], label="Historical Sales")
plt.plot(forecast_df["Month"], forecast_df["ARIMA_Forecast"], label="ARIMA
Forecast", linestyle="dashed", color="red")
plt.plot(forecast_df["Month"], forecast_df["Linear_Regression_Forecast"],
label="Linear Regression Forecast", linestyle="dashed", color="blue")
plt.title("Car Sales Forecast for Next 12 Months")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Save the forecast data to Excel
forecast_df.to_excel("car_sales_forecast.xlsx", index=False)
print("Forecast saved to car_sales_forecast.xlsx")