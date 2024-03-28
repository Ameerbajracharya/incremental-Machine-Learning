import time
import datetime

# Record the start time
start_time = time.time()
print("Started Time", datetime.datetime.now(), flush=True)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
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
X_subset = X.head(n=100000)
y_subset = y.head(n=100000)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)

model = SVR()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae, flush=True)
print("Mean Squared Error:", mse, flush=True)
print("R-squared Score:", r2, flush=True)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Time taken:", elapsed_time, "seconds", flush=True)

