import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from utils import load_and_process_data, train_models

# Page config
st.set_page_config(
    page_title="Football Player Prediction App",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main application header
st.title("⚽ Football Player Value & Potential Predictor")
st.markdown("""
This application predicts football player transfer values and classifies players
based on their potential using machine learning models. The models are trained on
historical player data from Kaggle datasets.
""")

# Main page content
st.header("About this Application")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transfer Value Prediction")
    st.markdown("""
    The application uses Linear Regression to predict a player's transfer value
    based on attributes like:
    - Age
    - Goals
    - Assists
    - Matches played
    - Overall rating
    """)

with col2:
    st.subheader("Player Classification")
    st.markdown("""
    Players are classified into three categories using Logistic Regression:
    - ⭐ **Star Player**: Potential rating of 85+
    - 👍 **Reliable Player**: Potential rating between 75-84
    - 👎 **Dud Player**: Potential rating below 75
    """)

# Load data if not already in session state
if 'data_loaded' not in st.session_state:
    with st.spinner("Downloading and processing datasets... This may take a moment."):
        try:
            # Load data and train models
            data, merged, X, y, X_class, y_class, reg_model, class_model = train_models()
            
            # Store in session state
            st.session_state['data'] = data
            st.session_state['merged'] = merged
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['X_class'] = X_class
            st.session_state['y_class'] = y_class
            st.session_state['reg_model'] = reg_model
            st.session_state['class_model'] = class_model
            st.session_state['data_loaded'] = True
            
            st.success("Data loaded and models trained successfully!")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please check your internet connection and try again. This app requires downloading datasets from Kaggle.")

# Navigation guide
st.header("Navigation")
st.markdown("""
Use the sidebar to navigate to different pages:
1. **Data Exploration**: Visualize and explore the football player datasets
2. **Model Performance**: Review the performance metrics of our prediction models
3. **Player Prediction**: Enter a player's stats to predict their transfer value and classification
""")

# Show a sample of the merged dataset
if 'merged' in st.session_state:
    st.header("Sample Data")
    st.dataframe(st.session_state['merged'].head())

# Footer
st.markdown("---")
st.markdown("Built with Streamlit (front end), Scikit-learn (machine learning model), and data from Kaggle")
st.markdown("Alphonse Otieno | © 2023 Football Player Prediction App")
