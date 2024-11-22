import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Preprocess Match Data
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

# Standardize team names
match['team1'] = match['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['team2'] = match['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['team1'] = match['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match['team2'] = match['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# Filter matches for valid teams and remove DLS-applied matches
match_df = match[
    (match['team1'].isin(teams)) & (match['team2'].isin(teams)) & (match['dl_applied'] == 0)
][['id', 'city', 'winner']]

# Merge deliveries with match data
delivery_df = delivery.merge(match_df, left_on='match_id', right_on='id')

# Filter for the second innings
delivery_df = delivery_df[delivery_df['inning'] == 2]

# Feature Engineering
delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs'].cumsum()
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
delivery_df['balls_left'] = 120 - (delivery_df['over'] * 6 + delivery_df['ball'])

# Calculate wickets left
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0).apply(lambda x: 1 if x != 0 else 0)
delivery_df['wickets'] = 10 - delivery_df.groupby('match_id')['player_dismissed'].cumsum()

# Calculate current run rate (CRR) and required run rate (RRR)
delivery_df['crr'] = delivery_df['current_score'] / ((120 - delivery_df['balls_left']) / 6)
delivery_df['rrr'] = delivery_df['runs_left'] / (delivery_df['balls_left'] / 6)

# Add match result
delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)

# Final dataset for training
final_df = delivery_df[
    ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr', 'result']
].dropna()
final_df = final_df[final_df['balls_left'] > 0]

# Split data into features and target
X = final_df.drop(columns='result')
y = final_df['result']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Preprocessing and Model Pipeline
trf = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city'])
    ],
    remainder='passthrough'
)

pipe = Pipeline(steps=[
    ('preprocessor', trf),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Train the model
pipe.fit(X_train, y_train)

# Evaluate the model
y_pred = pipe.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save the model
pickle.dump(pipe, open('pipe.pkl', 'wb'))

# Visualizing Match Progression
def match_progression(x_df, match_id, pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[match['ball'] == 6]
    temp_df = match[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr', 'rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] > 0]
    
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)
    
    target = temp_df['total_runs_x'].iloc[0]
    runs = list(temp_df['runs_left'])
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs[:-1]) - np.array(new_runs)
    wickets = list(temp_df['wickets'])
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[:temp_df.shape[0]]
    
    return temp_df, target

temp_df, target = match_progression(delivery_df, 50, pipe)
plt.figure(figsize=(18, 8))
plt.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='yellow', linewidth=3)
plt.plot(temp_df['end_of_over'], temp_df['win'], color='#00a65a', linewidth=4)
plt.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=4)
plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'])
plt.title(f'Target: {target}')
plt.show()
