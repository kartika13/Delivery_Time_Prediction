import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and feature names
model = pickle.load(open('delivery_time_model.pkl', 'rb'))
feature_names = pickle.load(open('features.pkl', 'rb'))

# Load data
df = pd.read_csv('Food_Delivery_Times.csv')

# App Title
st.title("🚚 Food Delivery Time Prediction & Analysis")

# Sidebar Menu
menu = ['EDA', 'Predict Delivery Time']
choice = st.sidebar.selectbox("Menu", menu)

# EDA Section
if choice == 'EDA':
    st.subheader("Exploratory Data Analysis")

    st.write("Sample Data:")
    st.dataframe(df.head())

    st.write("Delivery Time Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(df['Delivery_Time_min'], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("Vehicle Type Counts:")
    fig, ax = plt.subplots()
    sns.countplot(x='Vehicle_Type', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Weather Conditions Distribution:")
    fig, ax = plt.subplots()
    sns.countplot(x='Weather', data=df, ax=ax)
    st.pyplot(fig)

# Prediction Section
else:
    st.subheader("Estimate Delivery Time for an Order")

    Distance_km = st.slider('Distance (km)', 0.5, 30.0, 5.0)
    Preparation_Time_min = st.slider('Preparation Time (min)', 1, 60, 15)
    Courier_Experience_yrs = st.slider('Courier Experience (years)', 0, 20, 2)

    Weather = st.selectbox('Weather', df['Weather'].unique())
    Traffic_Level = st.selectbox('Traffic Level', df['Traffic_Level'].unique())
    Time_of_Day = st.selectbox('Time of Day', df['Time_of_Day'].unique())
    Vehicle_Type = st.selectbox('Vehicle Type', df['Vehicle_Type'].unique())

    if st.button('Estimate Delivery Time'):

        input_dict = {
            'Distance_km': Distance_km,
            'Preparation_Time_min': Preparation_Time_min,
            'Courier_Experience_yrs': Courier_Experience_yrs
        }

        # Add all expected features with default 0
        for col in feature_names:
            if col not in input_dict:
                input_dict[col] = 0

        # Activate relevant dummy columns
        for prefix, value in zip(['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type'], [Weather, Traffic_Level, Time_of_Day, Vehicle_Type]):
            col_name = f'{prefix}_{value}'
            if col_name in input_dict:
                input_dict[col_name] = 1

        # Final DataFrame with exact feature order
        input_df = pd.DataFrame([input_dict])[feature_names]

        predicted_time = model.predict(input_df)[0]

        st.write(f"⏱ Estimated Delivery Time: **{predicted_time:.2f} minutes**")
