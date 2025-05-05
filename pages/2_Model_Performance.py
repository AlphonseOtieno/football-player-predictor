import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, accuracy_score, classification_report 
# these metrics are used to evaluate the performance of the models
import joblib

# Page config: Setting up the Streamlit page for model performance
# This page displays the performance metrics of the trained models.
# This page allows users to analyze the performance of the machine learning models. Are they successful?
st.set_page_config(
    page_title="Model Performance - Football Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("🧪 Model Performance")
st.markdown("Analyze the performance of our machine learning models for transfer value prediction and player classification.")

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
    st.warning("Please return to the home page to load the data first.")
    st.stop()

# Get models and data from session state
X = st.session_state['X']
y = st.session_state['y']
X_class = st.session_state['X_class']
y_class = st.session_state['y_class']
reg_model = st.session_state['reg_model']
class_model = st.session_state['class_model']

# Evaluate regression model
st.header("Transfer Value Prediction Model")
st.subheader("Linear Regression Performance")

# Make predictions on the training data
y_pred = reg_model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mse)

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")

# Show model coefficients
st.subheader("Feature Importance")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg_model.coef_
})
coefficients['Absolute Coefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values('Absolute Coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis', ax=ax)
ax.set_title('Feature Coefficients in Linear Regression Model')
st.pyplot(fig)

# [Additional model evaluation code omitted for brevity]

# Evaluate classification model
st.header("Player Classification Model")
st.subheader("Logistic Regression Performance")

# Make predictions on the training data
y_class_pred = class_model.predict(X_class)

# Calculate metrics
accuracy = accuracy_score(y_class, y_class_pred)
class_report = classification_report(y_class, y_class_pred, output_dict=True)

# Display metrics
st.metric("Accuracy", f"{accuracy:.4f}")

# Display classification report
st.subheader("Classification Report")
class_df = pd.DataFrame(class_report).transpose()
st.dataframe(class_df)