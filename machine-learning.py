# ==============================================================
# SDG 13: CLIMATE ACTION 🌍
# PROJECT: Predicting CO₂ Emissions using Machine Learning
# APPROACH: Supervised Learning (Regression)
# ==============================================================
# Author: Leke Thaddeus Oladimeji
# Tools: Python, Pandas, Scikit-learn, Matplotlib
# ==============================================================

# --- Step 1: Import Required Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Step 2: Load Datasets ---
# For demo purposes, we’ll use sample datasets from the World Bank or Kaggle
# Replace the file paths/URLs with your actual dataset sources.

# Example file names (download and upload to Colab environment):
# 'co2_emissions.csv' -> EN.ATM.CO2E.PC
# 'energy_use.csv' -> EG.USE.PCAP.KG.OE
# 'gdp_per_capita.csv' -> NY.GDP.PCAP.CD
# 'urban_population.csv' -> SP.URB.TOTL.IN.ZS
# 'renewable_energy.csv' -> EG.FEC.RNEW.ZS

co2 = pd.read_csv('co2_emissions.csv')
energy = pd.read_csv('energy_use.csv')
gdp = pd.read_csv('gdp_per_capita.csv')
urban = pd.read_csv('urban_population.csv')
renew = pd.read_csv('renewable_energy.csv')

# --- Step 3: Inspect and Clean Data ---
print("CO₂ Data Preview:\n", co2.head())

# Ensure the data has columns like ['Country', 'Year', 'Value']
# Rename columns for consistency
for df in [co2, energy, gdp, urban, renew]:
    df.columns = ['Country', 'Year', df.columns[-1]]

# Merge datasets on 'Country' and 'Year'
data = co2.merge(energy, on=['Country', 'Year'])
data = data.merge(gdp, on=['Country', 'Year'])
data = data.merge(urban, on=['Country', 'Year'])
data = data.merge(renew, on=['Country', 'Year'])

# Rename final columns for clarity
data.columns = ['Country', 'Year', 'CO2', 'Energy_Use', 'GDP_per_Capita',
                'Urban_Pop', 'Renewable_Energy']

# --- Step 4: Handle Missing Values ---
data = data.dropna()
print("Cleaned Data Shape:", data.shape)

# --- Step 5: Define Features and Target ---
X = data[['Energy_Use', 'GDP_per_Capita', 'Urban_Pop', 'Renewable_Energy']]
y = data['CO2']

# --- Step 6: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 7: Normalize Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 8: Train Models ---

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lr = lin_reg.predict(X_test_scaled)

# 2. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

# --- Step 9: Evaluate Models ---
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\nModel: {model_name}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    return [mae, rmse, r2]

results = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'R2'])
results.loc[len(results)] = ['Linear Regression', *evaluate_model(y_test, y_pred_lr, "Linear Regression")]
results.loc[len(results)] = ['Random Forest', *evaluate_model(y_test, y_pred_rf, "Random Forest")]

print("\nModel Comparison:\n", results)

# --- Step 10: Visualize Results ---
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6)
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Random Forest: Actual vs Predicted CO₂ Emissions")
plt.grid(True)
plt.show()

# Feature Importance
importances = rf_reg.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(7,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title("Feature Importance in CO₂ Prediction")
plt.show()

# --- Step 11: Ethical Reflection Summary ---
print("""
ETHICAL REFLECTION 🌱
---------------------
• Bias Awareness: Developed nations have more complete data, which may bias predictions.
• Fairness: Models should be used to support sustainable growth policies in developing nations.
• Sustainability: Insights can guide investment in renewable energy to reduce emissions.
""")

# --- Step 12: Save Results (for GitHub submission) ---
results.to_csv('model_performance_summary.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\n✅ Project Completed Successfully!")
