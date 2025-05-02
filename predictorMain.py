import fastf1
import pandas as pd
import pickle

# Load trained model and encoders
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('le_driver.pkl', 'rb') as f:
    le_driver = pickle.load(f)
with open('le_team.pkl', 'rb') as f:
    le_team = pickle.load(f)

# Load 2025 Miami GP data (after qualifying)
session = fastf1.get_session(2025, 'Miami', 'Q')
session.load()
quali_results = session.results

# Prepare features (simplified; adjust based on actual data)
miami_data = []
for _, driver in quali_results.iterrows():
    # Placeholder for rolling stats (calculate from prior 2025 races)
    avg_finish = 10  # Replace with actual computation
    win_rate = 0    # Replace with actual computation
    row = {
        'DriverId': le_driver.transform([driver['DriverId']])[0],
        'TeamId': le_team.transform([driver['TeamId']])[0],
        'QualifyingPosition': driver['Position'],
        'PracticeAvgLapTime': session.laps[session.laps['DriverId'] == driver['DriverId']]['LapTime'].mean().total_seconds(),
        'Temperature': session.weather_data.get('AirTemp', 25) if hasattr(session, 'weather_data') else 25,
        'Rain': session.weather_data.get('Rainfall', False) if hasattr(session, 'weather_data') else False,
        'AvgFinishLast5': avg_finish,
        'WinRate': win_rate
    }
    miami_data.append(row)

miami_df = pd.DataFrame(miami_data)

# Predict probabilities
probs = rf_model.predict_proba(miami_df[rf_model.feature_names_in_])[:, 1]

# Add probabilities to DataFrame
miami_df['WinProbability'] = probs
miami_df['DriverId'] = le_driver.inverse_transform(miami_df['DriverId'])

# Select the predicted winner
predicted_winner = miami_df.loc[miami_df['WinProbability'].idxmax()]
print(f"Predicted winner of the 2025 Miami Grand Prix: {predicted_winner['DriverId']} "
      f"with probability {predicted_winner['WinProbability']:.2f}")
