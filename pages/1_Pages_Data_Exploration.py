import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_process_data # Import the function to load and process data from utils.py

# Page config
st.set_page_config(
    page_title="Data Exploration - Football Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("📊 Data Exploration")
st.markdown("Explore the football player datasets and visualize relationships between different attributes.")

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
    st.warning("Please return to the home page to load the data first.")
    st.stop()

# Get data from session state
merged = st.session_state['merged']

# Overview of the dataset
st.header("Dataset Overview")
st.write(f"Number of Players: {merged.shape[0]}")
st.write(f"Number of Features: {merged.shape[1]}")

# Data summary
st.subheader("Data Summary")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Key Features")
    key_features = ['Age', 'Goals', 'Assists', 'Matches', 'Overall rating', 'Potential', 'Value']
    st.dataframe(merged[key_features].describe())

with col2:
    st.markdown("### Player Classes Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = merged['Class'].value_counts()
    ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Player Classes')
    st.pyplot(fig)

# Visualizations
st.header("Visualizations")

# Visualization selector
viz_type = st.selectbox(
    "Select Visualization",
    ["Age vs Value", "Overall Rating vs Value", "Potential vs Value", 
     "Age Distribution", "Correlation Matrix", "Goals vs Value", "Assists vs Value"]
)

# Generate the selected visualization
if viz_type == "Age vs Value":
    st.subheader("Age vs Transfer Value")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=merged, x='Age', y='Value', hue='Class', palette='viridis', alpha=0.7, ax=ax)
    ax.set_title('Age vs Transfer Value by Player Class')
    ax.set_ylabel('Transfer Value (€)')
    st.pyplot(fig)
    
    st.markdown("""
    This plot shows the relationship between a player's age and their transfer value.
    Generally, players in their prime (mid-20s) tend to have higher transfer values,
    while both younger and older players may have lower values.
    """)

# [Additional visualizations code omitted for brevity]

# Value by age group
st.subheader("Average Transfer Value by Age Group")
merged['Age Group'] = pd.cut(merged['Age'], bins=[15, 20, 25, 30, 35, 40], labels=['16-20', '21-25', '26-30', '31-35', '36+'])
avg_value_by_age = merged.groupby('Age Group')['Value'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=avg_value_by_age, x='Age Group', y='Value', palette='viridis', ax=ax)
ax.set_title('Average Transfer Value by Age Group')
ax.set_ylabel('Average Transfer Value (€)')
st.pyplot(fig)

st.markdown("""
This bar chart shows how transfer values vary across different age groups.
Players in their prime years (typically early to mid-20s) often have the highest average transfer values.
""")