import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page config: Setting up the Streamlit page for player prediction
# This page allows users to input player statistics and predict their transfer value and classification.

# i'm building a model to predict the transfer value and potential of young football players, so I'll need 
# supervised machine learning models: LogisticRegression for classification (Star Player, Reliable, Dud) and
# LinearRegression for regression (Transfer Value in Euros) and/or potential numerical rating (1-100)

st.set_page_config(
    page_title="Player Prediction - Football Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("🔮 Player Value & Potential Prediction")
st.markdown("Enter a player's statistics to predict their transfer value and classification.")

# Check if data is loaded
if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
    st.warning("Please return to the home page to load the data first.")
    st.stop()

# Get models and data from session state
merged = st.session_state['merged']
reg_model = st.session_state['reg_model']
class_model = st.session_state['class_model']

# Create feature input form
st.header("Enter Player Statistics")

col1, col2 = st.columns(2)

with col1:
    player_name = st.text_input("Player Name (optional)")
    age = st.slider("Age", min_value=16, max_value=40, value=25, step=1)
    overall_rating = st.slider("Overall Rating", min_value=50, max_value=99, value=75, step=1)

with col2:
    goals = st.slider("Goals", min_value=0, max_value=50, value=10, step=1)
    assists = st.slider("Assists", min_value=0, max_value=50, value=5, step=1)
    matches = st.slider("Matches Played", min_value=1, max_value=100, value=30, step=1)

# Prediction button
predict_button = st.button("Predict", type="primary")

# Function to make predictions
def predict_player(features):
    # Reshape for single prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Predict transfer value
    value_prediction = reg_model.predict(features_array)[0]
    
    # Predict player class
    class_prediction = class_model.predict(features_array)[0]
    class_proba = class_model.predict_proba(features_array)[0]
    
    return value_prediction, class_prediction, class_proba

def format_value(value):
    if value >= 1_000_000:
        return f"€{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"€{value/1_000:.2f}K"
    else:
        return f"€{value:.2f}"

# Get class emoji
def get_class_emoji(class_name):
    if class_name == "Star":
        return "⭐ Star Player"
    elif class_name == "Reliable":
        return "👍 Reliable Player"
    else:
        return "👎 Dud Player"

# Make prediction when button is clicked
if predict_button:
    # Create feature array
    features = [age, goals, assists, matches, overall_rating]
    
    # Make prediction
    value_prediction, class_prediction, class_proba = predict_player(features)
    
    # Display results
    st.header("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transfer Value Prediction")
        formatted_value = format_value(value_prediction)
        st.markdown(f"<h1 style='text-align: center;'>{formatted_value}</h1>", unsafe_allow_html=True)
        
        # Compare with similar players
        st.subheader("Similar Players")
        
        # Find similar players based on features
        similar_players = merged.copy()
        similar_players['similarity'] = (
            (similar_players['Age'] - age)**2 +
            (similar_players['Goals'] - goals)**2 +
            (similar_players['Assists'] - assists)**2 +
            (similar_players['Matches'] - matches)**2 +
            (similar_players['Overall rating'] - overall_rating)**2
        )
        
        similar_players = similar_players.sort_values('similarity').head(5)
        st.dataframe(similar_players[['name', 'Age', 'Overall rating', 'Value', 'Class']])
    
    with col2:
        st.subheader("Player Classification")
        class_name = get_class_emoji(class_prediction)
        st.markdown(f"<h1 style='text-align: center;'>{class_name}</h1>", unsafe_allow_html=True)
        
        # Show classification probabilities
        class_names = reg_model.classes_ if hasattr(reg_model, 'classes_') else class_model.classes_
        proba_df = pd.DataFrame({
            'Class': class_names,
            'Probability': class_proba
        })
        
        # Create probability chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(x='Class', y='Probability', data=proba_df, palette='viridis', ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title('Classification Probability')
        
        # Add percentage labels on top of bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.02,
                f'{proba_df["Probability"].iloc[i]:.1%}',
                ha='center'
            )
        
        st.pyplot(fig)
    
    # Interpretation
    st.header("Interpretation")
    
    if class_prediction == "Star":
        st.markdown(f"""
        Based on the provided statistics, {player_name if player_name else "this player"} is predicted to be a **Star Player** with a transfer value of **{formatted_value}**.
        
        Star players have exceptional potential (85+ rating) and typically command high transfer fees. They are likely to become world-class talents and valuable assets for any club.
        
        **Key factors influencing this prediction:**
        - Overall rating of {overall_rating}
        - {goals} goals scored
        - {assists} assists provided
        - {matches} matches played
        - Age of {age} years
        """)
    
    # [Additional interpretation code for other classes omitted for brevity]

# Tips for accurate predictions
st.header("Tips for Accurate Predictions")
st.markdown("""
For the most accurate predictions, consider the following:

1. **Age Matters**: Generally, players between 23-28 years have the highest transfer values as they combine experience with future potential.

2. **Performance Stats**: Goals and assists significantly impact a player's valuation, especially for attacking players.

3. **Overall Rating**: This is one of the strongest predictors of both transfer value and player classification.

4. **Experience**: The number of matches played helps establish a player's consistency and reliability.

5. **Position Context**: Remember that different positions have different expected statistics. For example:
   - Strikers: High goals, moderate assists
   - Midfielders: Moderate goals, high assists
   - Defenders: Low goals and assists, but high match counts
""")