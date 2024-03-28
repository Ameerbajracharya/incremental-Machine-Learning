import time
# Record the start time
start_time = time.time()

from data_cleaning import data, X, y
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import r2_score
import xgboost as xgb
import numpy as np

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

# Initialize an empty list to store R-squared scores for each fold
r2_scores = []

# Perform KFold cross-validation
for train_index, test_index in kf.split(data):
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]

    # Split the data into features (X) and target (y)
    X_train, y_train = data_train.drop('soldPrice', axis=1), data_train['soldPrice']
    X_test, y_test = data_test.drop('soldPrice', axis=1), data_test['soldPrice']

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate R-squared for this fold
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Convert the list of R-squared scores to a numpy array
r2_scores = np.array(r2_scores)

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

# Print the R-squared scores for each fold
print("R-squared scores for each fold:", r2_scores)

# Calculate and print the mean R-squared score across all folds
mean_r2 = np.mean(r2_scores)
print("Mean R-squared score:", mean_r2)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Time taken:", elapsed_time, "seconds")