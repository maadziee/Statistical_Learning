import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Loading data
train = pd.read_csv("C:\\Users\\123\\Downloads\\train_ch.csv")
test = pd.read_csv("C:\\Users\\123\\Downloads\\test_ch.csv")

# Remove unnamed columns
train.drop(columns=[col for col in train.columns if 'Unnamed' in col], inplace=True)
test.drop(columns=[col for col in test.columns if 'Unnamed' in col], inplace=True)

# Define the features and the target
features_to_drop = ['v1', 'v5', 'v7', 'v8']
X_train = train.drop(features_to_drop + ['Y'], axis=1)
y_train = train['Y']

# Remove outliers based on z-scores
z_scores = np.abs(stats.zscore(X_train))
X_train = X_train[(z_scores < 3).all(axis=1)]
y_train = y_train[(z_scores < 3).all(axis=1)]

# Polynomial Features
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_transformer.fit_transform(X_train)

# Power Transformation
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_poly_scaled = power_transformer.fit_transform(X_train_poly)

# Linear Regression with Polynomial Features
linear_regression = LinearRegression()
linear_regression.fit(X_train_poly_scaled, y_train)
y_pred_linear = cross_val_predict(linear_regression, X_train_poly_scaled, y_train, cv=5)
rmse_linear = np.sqrt(mean_squared_error(y_train, y_pred_linear))

# Standard Scaler for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# KNN Regression
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
best_knn = grid_search.best_estimator_
y_pred_knn = cross_val_predict(best_knn, X_train_scaled, y_train, cv=5)
rmse_knn = np.sqrt(mean_squared_error(y_train, y_pred_knn))

# Results
print(f"Linear Regression with Polynomial Features - Cross-validated RMSE: {rmse_linear}")
print(f"KNN - Cross-validated RMSE: {rmse_knn}")

# Predicting on the test set
X_test = test.drop(features_to_drop, axis=1)
X_test_poly = poly_transformer.transform(X_test)
X_test_scaled = scaler.transform(X_test)  # For KNN
X_test_poly_scaled = power_transformer.transform(X_test_poly)  # For Linear Regression

# Linear Regression Predictions
linear_predictions = linear_regression.predict(X_test_poly_scaled)

# KNN Predictions
knn_predictions = best_knn.predict(X_test_scaled)

# Combine predictions into a DataFrame
predictions_df = pd.DataFrame({
    'Linear_Predictions': linear_predictions,
    'KNN_Predictions': knn_predictions
})

# Save the predictions to CSV
predictions_df.to_csv("C:\\Users\\123\\Downloads\\combined_predictions.csv", index=False)

print("Predictions for both models combined and saved successfully.")
