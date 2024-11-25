import streamlit as st
import pickle
import pandas as pd

# Team and City Options
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load Model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# App Title and Introduction
st.title('IPL Win Predictor')
st.markdown("""
This app predicts the win probabilities of IPL teams based on match data using a machine learning model.

### Instructions:
1. Select the batting and bowling teams.
2. Enter the current match details (target, score, overs, and wickets).
3. Click **Predict Probability** to see the results.
""")

# Input Features
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Prediction
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    try:
        result = pipe.predict_proba(input_df)
        win_prob = round(result[0][1] * 100, 2)
        loss_prob = round(result[0][0] * 100, 2)

        st.metric(label=f"{batting_team} Win Probability", value=f"{win_prob}%")
        st.metric(label=f"{bowling_team} Win Probability", value=f"{loss_prob}%")
    except Exception as e:
        st.error("Prediction failed. Please check your inputs and try again.")
        st.error(str(e))
