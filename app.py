import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the LSTM model
model = load_model('food_delivery_model.keras')

# Title of the Streamlit app
st.title("Food Delivery Time Prediction")

# Create a form for user input
st.subheader("Enter Details")
with st.form(key='prediction_form'):
    age = st.number_input("Age", min_value=0, step=1)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    distance = st.number_input("Distance (in meters)", min_value=0, step=1)
    submit_button = st.form_submit_button(label="Predict")

# Process prediction when the form is submitted
if submit_button:
    # Prepare input features
    features = np.array([[age, rating, distance]])
    features = features.reshape((features.shape[0], features.shape[1], 1))  # Reshape for LSTM

    # Make prediction
    prediction = model.predict(features)
    predicted_time = prediction[0][0]

    # Display prediction
    st.subheader("Prediction")
    st.write(f"Predicted Delivery Time: {predicted_time:.2f} minutes")
