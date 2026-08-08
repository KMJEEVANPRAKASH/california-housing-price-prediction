# California Housing Price Prediction

A Machine Learning project that predicts California house values
using Linear Regression.

## Project Overview

This project uses the California Housing dataset and applies
Exploratory Data Analysis and Linear Regression to predict
median house values.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Workflow

1. Load California Housing dataset
2. Data exploration
3. Statistical analysis
4. Missing-value analysis
5. Data visualization
6. Correlation analysis
7. Train-test split
8. Linear Regression model
9. Model evaluation
10. Prediction visualization

## Evaluation Metrics

- MAE
- RMSE
- R² Score

## Dataset

California Housing dataset provided through Scikit-learn.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
# Example: Loading a CSV file from Google Drive
# Make sure to replace this path with the actual path to your file
try:
    df_from_drive = pd.read_csv('/content/drive/MyDrive/your_folder/your_data.csv')
    print('Data loaded successfully from Google Drive:')
    display(df_from_drive.head())
except FileNotFoundError:
    print("Error: File not found. Please check the path to your data file in Google Drive.")
    print("Example path: '/content/drive/MyDrive/my_data_folder/my_file.csv'")
except Exception as e:
    print(f"An error occurred while loading data from Google Drive: {e}")

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print('Original California Housing DataFrame head:')
display(df.head())

print('\nDataFrame Info:')
df.info()

print('\nDataFrame Description:')
display(df.describe())

print('\nDataFrame Shape:')
print(df.shape)

print('\nMissing Values:')
print(df.isnull().sum())

# Histograms
df.hist(figsize=(15,10))
plt.suptitle('Histograms of California Housing Features')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title('Correlation Heatmap of California Housing Features')
plt.show()

# Pairplot
sns.pairplot(df[['MedInc','HouseAge','AveRooms','MedHouseVal']])
plt.suptitle('Pairplot of Selected California Housing Features', y=1.02)
plt.show()

# Prepare data for modeling
X = df.drop("MedHouseVal",axis=1)
y = df["MedHouseVal"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train,y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test,y_pred)
print(f"\nMean Absolute Error (MAE): {mae:.2f}")

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

r2 = r2_score(y_test,y_pred)
print(f"R-squared (R2): {r2:.2f}")

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Residual Plot
residuals = y_test-y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred,residuals, alpha=0.5)
plt.axhline(0,color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
