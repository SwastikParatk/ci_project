import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('pipe.pkl', 'rb'))

# Input form
st.title("Cricket Match Winning Probability Predictor")

# Input fields
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Delhi', 'Jaipur'
]

st.sidebar.header("Input Match Details")
batting_team = st.sidebar.selectbox("Batting Team", teams)
bowling_team = st.sidebar.selectbox("Bowling Team", teams)
host_city = st.sidebar.selectbox("Host City", cities)
target = st.sidebar.number_input("Target Score", min_value=1, step=1)
overs_completed = st.sidebar.slider("Overs Completed", 0.0, 20.0, step=0.1)
wickets_out = st.sidebar.slider("Wickets Out", 0, 10, step=1)

# Predict button
if st.sidebar.button("Predict Winning Probability"):
    balls_left = int((20 - overs_completed) * 6)
    runs_left = target
    crr = (target - runs_left) / overs_completed if overs_completed > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    input_data = np.array([[
        batting_team,
        bowling_team,
        host_city,
        runs_left,
        balls_left,
        10 - wickets_out,
        target,
        crr,
        rrr
    ]])

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]
    bowling_win_prob = probabilities[0] * 100
    batting_win_prob = probabilities[1] * 100

    # Display results
    st.write(f"### Batting Team Winning Probability: {batting_win_prob:.2f}%")
    st.write(f"### Bowling Team Winning Probability: {bowling_win_prob:.2f}%")
