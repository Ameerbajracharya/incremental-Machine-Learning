import pandas as pd
import numpy as np
import joblib
import datetime
import locale, math
from data_cleaning import minDate

# Load the trained model
model = joblib.load('xgboost_model.pkl')

# Load the target encoder
target_encoder = joblib.load('target_encoder.pkl')

# Define a function to preprocess user inputs
def preprocess_inputs(address, state, postcode, suburb, ptype):
    # Create a DataFrame with user inputs
    data = pd.DataFrame({
        'address': [address],
        'suburb': [suburb],
        'state': [state],
        'postcode': [postcode],
        'pType': [ptype]
    })
    # Perform target encoding for categorical columns
    data_encoded = target_encoder.transform(data)

    additional_data = pd.DataFrame({
        'soldDate': [(pd.to_datetime(datetime.datetime.now()) - minDate).days],
        'beds': [0],
        'bathrooms': [0],
        'carSpaces': [0]
    })

      # Concatenate additional_data with data_encoded
    data_encoded = pd.concat([additional_data, data_encoded], axis=1)

    # Define the expected order of columns
    expected_columns = ['soldDate', 'beds', 'bathrooms', 'carSpaces', 'address', 'state', 'postcode', 'suburb', 'pType']

    # Reorder the columns of data_encoded to match the expected order
    data_encoded = data_encoded.reindex(columns=expected_columns)

    return data_encoded

# Define a function to predict property price
def predict_price(model, data):
    # Make predictions using the loaded model
    predicted_price = model.predict(data)
    return predicted_price

# Main function to get user input and make predictions
def main():
    # Get user input
    address = input("Enter the address: ")
    suburb = input("Enter the suburb: ")
    # state = input("Enter the state: ")
    postcode = input("Enter the postcode: ")
    ptype = input("Enter the property type: ")

    # Preprocess user inputs
    data = preprocess_inputs(address, "nsw", postcode, suburb, ptype)

    # Predict property price
    predicted_price = predict_price(model, data)
    locale.setlocale(locale.LC_ALL, 'en_AU.UTF-8')

    predicted_price_scalar = predicted_price[0]  # Assuming the predicted price is the first element of the array

    # Round up the predicted price
    rounded_predicted_price = math.ceil(predicted_price_scalar/2)

    # Format the rounded predicted price as currency
    formatted_currency = locale.currency(rounded_predicted_price, grouping=True)

    print("Predicted Price:", formatted_currency)

if __name__ == "__main__":
    main()
