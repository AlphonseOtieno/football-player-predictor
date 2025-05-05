import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

def load_and_process_data():
    """
    Load and process the football player datasets from kaggle.
    This function downloads the datasets, cleans the data, and merges them.
    """
    # Download datasets from Kaggle
    path = kagglehub.dataset_download("dhruvpjariwala/football-players-transfer-value-prediction")
    path2 = kagglehub.dataset_download("abdulmalik1518/football-players-datasets-2015-2024")
    
    # Load datasets
    csv_file = f"{path}/football_players_value_dataset.csv"
    data = pd.read_csv(csv_file)
    data.drop(columns=['Unnamed: 64'], errors='ignore', inplace=True) #removing unnecessary column with no helpful data
    
    csv_file2 = f"{path2}/Latest Football  Players 2024 Data.csv"
    data2 = pd.read_csv(csv_file2)
    
    # Clean player names in both datasets
    data['name'] = data['name'].str.lower()
    data['name'] = data['name'].str.replace('.', '')
    data['name'] = data['name'].str.split(' ').str[1:]
    data['name'] = data['name'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    
    data2['Players'] = data2['Players'].str.lower()
    data2['Players'] = data2['Players'].str.replace('.', '')
    data2['Players'] = data2['Players'].str.split(' ').str[1:]
    data2['Players'] = data2['Players'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
    
    # Merge datasets
    merged = pd.merge(data, data2, left_on="name", right_on="Players", how="inner")
    
    # Clean overall rating and potential
    merged["Overall rating"] = merged["Overall rating"].astype(str).str.extract(r'(\d+)').astype(float)
    merged["Potential"] = merged["Potential"].astype(str).str.extract(r'(\d+)').astype(float)
    
    # Drop rows with missing values in key columns
    merged = merged.dropna(subset=["Overall rating", "Potential"])
    
    # Clean value column
    merged["Value"] = merged["Value"].astype(str).replace('€', '', regex=True)
    # Extract numeric part and handle K/M suffixes
    merged["Value"] = merged["Value"].apply(
        lambda x: float(x.replace('M', '')) * 1000000 if 'M' in x else
        (float(x.replace('K', '')) * 1000 if 'K' in x else float(x))
    )
    
    # Classify players based on potential
    merged["Class"] = merged["Potential"].astype(float).apply(classify_player)
    
    return data, data2, merged

def classify_player(potential):
    """
    Classify a player based on their potential rating
    """
    if potential >= 85:
        return "Star"  # Star Player
    elif potential >= 75:
        return "Reliable"  # Reliable Player
    else:
        return "Dud"  # Dud Player

def train_models():
    """
    Load data and train the regression and classification models
    """
    # Load and process data
    data, data2, merged = load_and_process_data()
    
    # Target variable for regression: Transfer Value
    y = merged["Value"]
    
    # Features for both models
    X = merged[["Age", "Goals", "Assists", "Matches", "Overall rating"]]
    
    # Train regression model
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    
    # Target variable for classification: Player class
    y_class = merged["Class"]
    
    # Train classification model
    X_class = X.copy()  # Same features as regression
    class_model = LogisticRegression(max_iter=1000)
    class_model.fit(X_class, y_class)
    
    return data, merged, X, y, X_class, y_class, reg_model, class_model

