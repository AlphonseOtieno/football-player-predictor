import kagglehub

"""
this is an unsuccessful attempt to build a the ML model
"""

# Download latest version of the datasets from Kaggle

path = kagglehub.dataset_download("dhruvpjariwala/football-players-transfer-value-prediction")
path2 = kagglehub.dataset_download("abdulmalik1518/football-players-datasets-2015-2024")

# ------------------------------------------------------------------------------
# SECTION 1: Import Libraries

import os # For file path operations
import io # For file operations
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For data visualization
import seaborn as sns # For data visualization

# ------------------------------------------------------------------------------
# i'm building a model to predict the transfer value and potential of young football players, so I'll need 
# supervised machine learning models: LogisticRegression for classification (Star Player, Reliable, Dud) and
# LinearRegression for regression (Transfer Value in Euros) and/or potential numerical rating (1-100)

from sklearn.linear_model import LinearRegression # For regression
from sklearn.linear_model import LogisticRegression # For classification


csv_file = f"{path}/football_players_value_dataset.csv"  # Replace 'filename.csv' with the actual CSV file name
data = pd.read_csv(csv_file)

data.drop(columns=['Unnamed: 64'], errors='ignore', inplace=True) # Drop the 'Unnamed: 64' -- cleaning

csv_file2 = f"{path2}/Latest Football  Players 2024 Data.csv"  # Replace 'filename.csv' with the actual CSV file name
data2 = pd.read_csv(csv_file2)

# Display the first few rows of the dataset
#print(data.head())
#print(data2.head()) #both successfully print!

#print(data.columns) #to see the columns in the dataset
#print(data2.columns) #to see the columns in the dataset


# names are in the format 'FirstInitial. Last position1 position2' or 
# 'First Last position1 position2' eg 'A. Younes LM CAM' or 
# R. Sessegnon LB LWB
data['name'] = data['name'].str.lower() #lowercase
data['name'] = data['name'].str.replace('.', '') #remove periods
#remove the first name from the name column
data['name'] = data['name'].str.split(' ').str[1:] #remove first name, keeping only last name
#remove everything after the first name, which are positions eg. st or lw:
data['name'] = data['name'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')  # Keep only the last name, which is the first element of the list


#print(data.head())

#do the same for data2
data2['Players'] = data2['Players'].str.lower() #lowercase
data2['Players'] = data2['Players'].str.replace('.', '') #remove periods, if any
data2['Players'] = data2['Players'].str.split(' ').str[1:] #remove first name, keeping only last name
data2['Players'] = data2['Players'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')  # Keep only the last name, which is the first element of the list
#print(data2.head())

#now the names are in the same format.

# Merge
merged = pd.merge(data, data2, left_on="name", right_on="Players", how="inner")
#print(merged.head())

# Clean 'Overall rating' and 'Potential' columns
merged["Overall rating"] = merged["Overall rating"].astype(str).str.extract(r'(\d+)').astype(float)
merged["Potential"] = merged["Potential"].astype(str).str.extract(r'(\d+)').astype(float)

# Drop rows with invalid or missing values
merged = merged.dropna(subset=["Overall rating", "Potential"])

#error with Value column, need to clean it. in file, eg. €46M
# maybe make it into a float, by removing the euro sign and converting to float
merged["Value"] = merged["Value"].str.replace("€", "").str.replace("M", "").str.replace("K", "").str.replace(",", "").astype(float)
# Convert 'K' to thousands and 'M' to millions
merged["Value"] = merged["Value"].apply(lambda x: x * 1000 if "K" in str(x) else x)
merged["Value"] = merged["Value"].apply(lambda x: x * 1000000 if "M" in str(x) else x)


# Target variable: Transfer Value in Euros
y = merged["Value"]

# Features (you can expand this)
X = merged[["name", "Age", "Goals", "Assists", "Matches", "Overall rating", "Seasons"]]
print(X.head())
# you can see different versions of the same player in the dataset, which is good for training the model
# as we see the player develop over time. or maybe, im not exactly sure yet.
X = X.drop(columns=["name", "Seasons"]) # Drop the 'name' and 'Seasons' columns, not useful for training

#-------------------------------------------------------------------
# Training the Regression Model
model = LinearRegression()
model.fit(X, y)

# Optional: Evaluate on training data
predictions = model.predict(X)

from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y, predictions)) # Mean Squared Error, a measure of how close the predictions are to the actual values. A good model will have a low MSE like 0.1 or below.
print("R^2 Score:", r2_score(y, predictions)) # R^2 Score, a measure of how well the model explains the variance in the data. A good model will have a high R^2 score like 0.8 or above.

# Training the Classification Model

def classify_player(potential):
    if potential >= 85:
        return "Star" # Star Player
    elif potential >= 75:
        return "Reliable" # Reliable Player
    else:
        return "Dud" # Dud Player

merged["Class"] = merged["Potential"].astype(float).apply(classify_player)
y_class = merged["Class"]

X_class = X.copy()  # reuse same features as before
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_class, y_class)