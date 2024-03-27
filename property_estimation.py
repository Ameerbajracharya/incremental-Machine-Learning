import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

# Define dtype mapping for reading CSV
dtype_mapping = {
    "soldDate": 'str',
    "state": 'str',
    "postcode": 'str',
    "suburb": 'str',
    "address": 'str',
    "soldPrice": 'str',
    "beds": 'int',
    "bathrooms": 'int',
    "carSpaces": 'int',
    "pType": 'str'
}

# Load the data into a pandas DataFrame
data = pd.read_csv('kaggle/property_data.csv', dtype=dtype_mapping)

# Convert 'soldDate' to datetime and days since the earliest date
data['soldDate'] = pd.to_datetime(data['soldDate'])
data['soldDate'] = (data['soldDate'] - data['soldDate'].min()).dt.days

# Remove rows with invalid 'soldPrice'
data = data[~data['soldPrice'].isin(['Contact Agent', 'Contact agent'])]
mask = ~data['soldPrice'].str.contains(r'[-\.\.]', na=False)
data = data[mask]

# Reset the index of the DataFrame
data.reset_index(drop=True, inplace=True)

# Remove the dollar sign and commas from 'soldPrice' and convert to float
data['soldPrice'] = data['soldPrice'].str.replace('[$,]', '', regex=True).astype(float)

# Handle missing values
data = data.dropna(subset=['soldPrice', 'beds', 'bathrooms', 'carSpaces'])

# Define columns to be target encoded
columns_to_encode = ['address', 'state', 'postcode', 'suburb', 'pType']

# Use target encoding for the specified columns
target_encoder = TargetEncoder(cols=columns_to_encode)
data_encoded = target_encoder.fit_transform(data[columns_to_encode], data['soldPrice'])

# Replace original columns with encoded columns in the DataFrame
data.drop(columns_to_encode, axis=1, inplace=True)
data = pd.concat([data, data_encoded], axis=1)

# Split the data into features (X) and target (y)
X = data.drop('soldPrice', axis=1)
y = data['soldPrice']

# Define hyperparameters for tuning
param_grid = {
    'learning_rate': [0.05, 0.06, 0.001, 0.002],
    'max_depth': [11, 15, 13, 14],
    'n_estimators': [120, 130, 140, 110]
}

# Initialize XGBoost regressor
model = xgb.XGBRegressor()

# Perform grid search with cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
grid_search.fit(X, y)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Negative MAE:", grid_search.best_score_)

# Use the best model found by grid search
best_model = grid_search.best_estimator_

# Evaluate the best model using cross-validation
cv_results = cross_val_score(best_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
print("Cross-Validation Results (Negative MAE):", cv_results)

# Convert negative MAE scores to positive and calculate mean
cv_results = -cv_results
mean_cv_mae = np.mean(cv_results)
print("Mean Cross-Validation MAE:", mean_cv_mae)
