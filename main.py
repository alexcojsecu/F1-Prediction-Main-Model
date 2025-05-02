import fastf1
import pandas as pd
from datetime import datetime

# Define seasons and races (simplified example: all races from 2020-2024)
seasons = range(2020, 2025)
race_data = []

# Collect data for each race
for year in seasons:
    # Load event schedule for the year
    schedule = fastf1.get_event_schedule(year)
    for _, event in schedule.iterrows():
        race_name = event['EventName']
        try:
            # Load race session
            race = fastf1.get_session(year, race_name, 'R')
            race.load()
            race_results = race.results

            # Load qualifying session
            quali = fastf1.get_session(year, race_name, 'Q')
            quali.load()
            quali_results = quali.results

            # Get weather data (if available)
            weather = race.weather_data if hasattr(race, 'weather_data') else {}

            # Process data for each driver
            for driver_id in race_results['DriverId']:
                driver_race = race_results[race_results['DriverId'] == driver_id].iloc[0]
                driver_quali = quali_results[quali_results['DriverId'] == driver_id].iloc[0]

                # Calculate rolling stats (simplified here; in practice, compute from prior races)
                past_races = [r for r in race_data if r['DriverId'] == driver_id][-5:]
                avg_finish = sum(r['Position'] for r in past_races) / len(past_races) if past_races else 10
                win_rate = sum(1 for r in past_races if r['Position'] == 1) / len(past_races) if past_races else 0

                # Compile row data
                row = {
                    'Year': year,
                    'RaceName': race_name,
                    'DriverId': driver_id,
                    'TeamId': driver_race['TeamId'],
                    'QualifyingPosition': driver_quali['Position'] if pd.notna(driver_quali['Position']) else 20,
                    'PracticeAvgLapTime': race.laps[race.laps['DriverId'] == driver_id]['LapTime'].mean().total_seconds() if not race.laps.empty else None,
                    'Temperature': weather.get('AirTemp', None),
                    'Rain': weather.get('Rainfall', False),
                    'AvgFinishLast5': avg_finish,
                    'WinRate': win_rate,
                    'Position': driver_race['Position'] if pd.notna(driver_race['Position']) else 20,
                    'Won': 1 if driver_race['Position'] == 1 else 0
                }
                race_data.append(row)
        except Exception as e:
            print(f"Error loading {race_name} {year}: {e}")

# Convert to DataFrame
df = pd.DataFrame(race_data)
df.to_csv('f1_race_data.csv', index=False)
print("Data collection complete. Saved to f1_race_data.csv")