import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load Data
df = pd.read_csv('Food_Delivery_Times.csv')

# Select Features
features = [
    'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs'
]

# Convert categorical to dummies
df = pd.get_dummies(df, columns=['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'], drop_first=True)

X = df[features + [col for col in df.columns if col.startswith(('Weather_', 'Traffic_Level_', 'Time_of_Day_', 'Vehicle_Type_'))]]
y = df['Delivery_Time_min']

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save Model
with open('delivery_time_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save Feature Names
with open('features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained and saved successfully.")
