# ============================================================
# California Housing Price Prediction
# Model Training
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

housing = fetch_california_housing()

X = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

y = pd.Series(
    housing.target,
    name="MedHouseValue"
)

print("Dataset loaded successfully!")
print("Features:", X.shape)
print("Target:", y.shape)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(model_name, y_test, y_pred):

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("MAE  :", mae)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("R²   :", r2)

    return mae, mse, rmse, r2


# ============================================================
# 3. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(X_test)

linear_mae, linear_mse, linear_rmse, linear_r2 = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)


# ============================================================
# 4. RIDGE REGRESSION
# ============================================================

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train,
    y_train
)

ridge_pred = ridge_model.predict(X_test)

ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_pred
)


# ============================================================
# 5. LASSO REGRESSION
# ============================================================

lasso_model = Lasso(
    alpha=0.001
)

lasso_model.fit(
    X_train,
    y_train
)

lasso_pred = lasso_model.predict(X_test)

lasso_mae, lasso_mse, lasso_rmse, lasso_r2 = evaluate_model(
    "Lasso Regression",
    y_test,
    lasso_pred
)


# ============================================================
# 6. RANDOM FOREST REGRESSOR
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)

rf_mae, rf_mse, rf_rmse, rf_r2 = evaluate_model(
    "Random Forest Regressor",
    y_test,
    rf_pred
)


# ============================================================
# 7. GRADIENT BOOSTING REGRESSOR
# ============================================================

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train,
    y_train
)

gb_pred = gb_model.predict(X_test)

gb_mae, gb_mse, gb_rmse, gb_r2 = evaluate_model(
    "Gradient Boosting Regressor",
    y_test,
    gb_pred
)


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest Regressor",
        "Gradient Boosting Regressor"
    ],

    "MAE": [
        linear_mae,
        ridge_mae,
        lasso_mae,
        rf_mae,
        gb_mae
    ],

    "MSE": [
        linear_mse,
        ridge_mse,
        lasso_mse,
        rf_mse,
        gb_mse
    ],

    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse,
        rf_rmse,
        gb_rmse
    ],

    "R2 Score": [
        linear_r2,
        ridge_r2,
        lasso_r2,
        rf_r2,
        gb_r2
    ]
})


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.sort_values(
        by="R2 Score",
        ascending=False
    ).to_string(index=False)
)
