import pandas as pd
from category_encoders import TargetEncoder
import joblib

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
minDate = data['soldDate'].min()
data['soldDate'] = (data['soldDate'] - minDate).dt.days

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

# Save the target encoder
# joblib.dump(target_encoder, 'target_encoder.pkl')

# Replace original columns with encoded columns in the DataFrame
data.drop(columns_to_encode, axis=1, inplace=True)
data = pd.concat([data, data_encoded], axis=1)

# Split the data into features (X) and target (y)
X = data.drop('soldPrice', axis=1)
y = data['soldPrice']

