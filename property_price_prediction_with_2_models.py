import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
from category_encoders import TargetEncoder
from sklearn.metrics import mean_absolute_error

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
data = pd.read_csv('property_data.csv', dtype=dtype_mapping)

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

# Define pipelines for XGBoost and SVR
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('xgb', xgb.XGBRegressor())    # XGBoost regressor
])

svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('svr', SVR())                 # SVR regressor
])

# Define parameter grids for hyperparameter tuning
xgb_param_grid = {
    'xgb__learning_rate': [0.1, 0.05, 0.01],
    'xgb__max_depth': [3, 5, 7],
    'xgb__n_estimators': [100, 200, 300]
}

svr_param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__gamma': ['scale', 'auto'],
}

# Perform grid search with cross-validation for XGBoost
kf = KFold(n_splits=5, shuffle=True, random_state=42)

xgb_grid_search = GridSearchCV(xgb_pipeline, xgb_param_grid, cv=kf, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
xgb_grid_search.fit(X, y)

# Print best parameters and best score for XGBoost
print("XGBoost Best Parameters:", xgb_grid_search.best_params_)
print("XGBoost Best Negative MAE:", xgb_grid_search.best_score_)

# Perform grid search with cross-validation for SVR
svr_grid_search = GridSearchCV(svr_pipeline, svr_param_grid, cv=kf, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)
svr_grid_search.fit(X, y)

# Print best parameters and best score for SVR
print("SVR Best Parameters:", svr_grid_search.best_params_)
print("SVR Best Negative MAE:", svr_grid_search.best_score_)

# Calculate mean cross-validation MAE for XGBoost
xgb_cv_mae = -xgb_grid_search.best_score_
print("Mean Cross-Validation MAE for XGBoost:", xgb_cv_mae)

# Calculate mean cross-validation MAE for SVR
svr_cv_mae = -svr_grid_search.best_score_
print("Mean Cross-Validation MAE for SVR:", svr_cv_mae)

