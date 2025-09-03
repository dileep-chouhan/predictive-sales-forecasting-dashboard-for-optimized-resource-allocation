import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_periods = 120
dates = pd.date_range(start='2022-01-01', periods=num_periods, freq='M')
sales = 1000 + np.random.randn(num_periods) * 100 + np.arange(num_periods) * 50 #Trend + Seasonality + Noise
advertising_spend = 100 + np.random.randn(num_periods) * 20 + np.arange(num_periods) * 5 #Trend + Noise
promotions = np.random.randint(0, 2, num_periods) #Binary: 0 or 1
df = pd.DataFrame({'Date': dates, 'Sales': sales, 'Advertising': advertising_spend, 'Promotions': promotions})
# --- 2. Data Cleaning and Feature Engineering ---
# In a real-world scenario, this section would include handling missing values, outliers, etc.
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
# --- 3. Time Series Analysis and Forecasting ---
# Prepare data for time series analysis
y = df['Sales']
X = df[['Month', 'Year', 'Advertising', 'Promotions']]
X = sm.add_constant(X) # Add a constant term
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
# Forecast future sales (example: next 12 months)
future_dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
future_df = pd.DataFrame({'Date': future_dates, 'Month': future_dates.dt.month, 'Year': future_dates.dt.year, 'Advertising': 200, 'Promotions': 0}) # Assume constant advertising and no promotions
future_df = sm.add_constant(future_df)
forecasted_sales = results.predict(future_df)
future_df['Forecasted_Sales'] = forecasted_sales
# --- 4. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Sales'], label='Actual Sales')
plt.plot(future_df['Date'], future_df['Forecasted_Sales'], label='Forecasted Sales', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Forecast')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sales_forecast.png')
print("Plot saved to sales_forecast.png")
plt.figure(figsize=(10,6))
sns.regplot(x='Advertising', y='Sales', data=df)
plt.title('Advertising Spend vs. Sales')
plt.savefig('adv_sales.png')
print("Plot saved to adv_sales.png")