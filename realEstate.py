import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


import os
for dirname, _, filenames in os.walk('/kaggle/input/australian-housing-data-1000-properties-sampled'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("/workspaces/incremental-Machine-Learning/kaggle/input/australian-housing-data-1000-properties-sampled/RealEstateAU_1000_Samples.csv")

df.drop(columns = ['latitude' , 'longitude' ,'building_size' , 'land_size' , 'preferred_size' , 'open_date'] , axis = 1, inplace = True)
df.drop(columns = ['TID' , 'RunDate' , 'phone', 'breadcrumb', 'address', 'category_name' , 'location_number'] , axis = 1, inplace = True)

def df_clean(df, column_name):
    df[column_name] = df[column_name].str.replace(r'^.*?\$', '', regex=True)


df_clean(df, 'price')
df_clean(df, 'location_name')

matching_values = (df['price'] == df['location_name']).sum()

df = df.drop('location_name' , axis = 1)
df.dropna(subset =['bedroom_count' , 'bathroom_count' , 'parking_count' , 'address_1'] ,inplace = True)
df.isnull().sum()

df['price'] = df['price'].str.replace(',', '', regex=True)
df = df[df['price'].str.isnumeric()]
df.reset_index(drop=True, inplace=True)
df['price'] = df['price'].astype(int)

df.drop('index' , axis=1,inplace=True)
df.drop('location_type' , axis=1, inplace=True)


plt.figure(figsize=(12,5))
ax =sns.countplot(x='property_type' , data= df )
plt.xlabel('Property Type')
plt.xticks(rotation = 90)
plt.ylabel('Count')
plt.title('property market analysis ')

def add_value_labels(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')


add_value_labels(ax)

plt.show()

max_prices = df.groupby('property_type')['price'].transform('max')
min_prices = df.groupby('property_type')['price'].transform('min')

plt.figure(figsize=(10, 6))
bars = plt.bar(df['property_type'], df['price'])
plt.xlabel('Property Type')
plt.xticks(rotation=90)
plt.ylabel('Price')

for bar, max_val, min_val in zip(bars, max_prices, min_prices):
    if bar.get_height() == max_val:
        plt.annotate(f'Max: {max_val}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2),
                     xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8, color='blue')
    elif bar.get_height() == min_val:
        plt.annotate(f'Min: {min_val}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() - 10),
                     xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color='red')

plt.title('Price by Property Type')
plt.show()