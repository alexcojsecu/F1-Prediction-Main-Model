import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load data
df = pd.read_csv('f1_race_data.csv')

# Preprocessing
# Encode categorical variables
le_driver = LabelEncoder()
le_team = LabelEncoder()
df['DriverId'] = le_driver.fit_transform(df['DriverId'])
df['TeamId'] = le_team.fit_transform(df['TeamId'])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
features = ['DriverId', 'TeamId', 'QualifyingPosition', 'PracticeAvgLapTime', 
            'Temperature', 'Rain', 'AvgFinishLast5', 'WinRate']
df[features] = imputer.fit_transform(df[features])

# Define features and target
X = df[features]
y = df['Won']

# Split data (2020-2023 for training, 2024 for validation)
train_df = df[df['Year'] < 2024]
val_df = df[df['Year'] == 2024]
X_train = train_df[features]
y_train = train_df['Won']
X_val = val_df[features]
y_val = val_df['Won']

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and encoders
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('le_driver.pkl', 'wb') as f:
    pickle.dump(le_driver, f)
with open('le_team.pkl', 'wb') as f:
    pickle.dump(le_team, f)

print("Model training complete. Model and encoders saved.")