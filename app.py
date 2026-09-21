import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('logi.sav')

st.title('Delivery Delay Prediction')

st.write("Enter the features to predict if there will be a delivery delay:")

# Input fields for features, using X.columns for reference
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=12.0, help='Distance of the delivery in km')
traffic_congestion = st.slider('Traffic Congestion', min_value=1, max_value=5, value=2, help='Level of traffic congestion (1=low, 5=high)')
weather_condition = st.slider('Weather Condition', min_value=1, max_value=5, value=1, help='Weather severity (1=clear, 5=severe)')
delivery_slot = st.slider('Delivery Slot', min_value=1, max_value=3, value=2, help='Preferred delivery time slot (1, 2, or 3)')
driver_experience = st.slider('Driver Experience (years)', min_value=0, max_value=20, value=4, help='Years of experience of the driver')
num_stops = st.slider('Number of Stops', min_value=0, max_value=10, value=3, help='Number of stops the driver has to make')
vehicle_age = st.slider('Vehicle Age (years)', min_value=0, max_value=15, value=3, help='Age of the delivery vehicle')
road_condition_score = st.slider('Road Condition Score', min_value=1, max_value=5, value=3, help='Quality of the roads (1=poor, 5=excellent)')
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=12.0, help='Weight of the package')
fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, value=10.0, help='Fuel efficiency of the vehicle')
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=120, help='Time taken to process the package at the warehouse')

# Create a DataFrame from inputs
# The order of columns must match the training data (X.columns from cell ak3XdJ-sD3vL)
input_data = pd.DataFrame([[
    delivery_distance,
    traffic_congestion,
    weather_condition,
    delivery_slot,
    driver_experience,
    num_stops,
    vehicle_age,
    road_condition_score,
    package_weight,
    fuel_efficiency,
    warehouse_processing_time
]], columns=['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
       'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
       'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
       'Warehouse_Processing_Time'])

# Make prediction
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Result:')
    if prediction == 1:
        st.error(f"Delivery will likely be **Delayed** (Probability of delay: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"Delivery will likely **NOT be Delayed** (Probability of no delay: {prediction_proba[0]*100:.2f}%)")
