import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Food Delivery Time Prediction")

# Load model with error handling
try:
    logger.info("Loading model...")
    model = load_model('food_delivery_model.keras')
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    logger.error(f"Model loading error: {str(e)}")
    st.stop()

# Input form
st.subheader("Enter Details")
with st.form(key='prediction_form'):
    age = st.number_input("Age", min_value=0, step=1)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    distance = st.number_input("Distance (in meters)", min_value=0, step=1)
    submit_button = st.form_submit_button(label="Predict")

# Prediction
if submit_button:
    try:
        logger.info(f"Input features: age={age}, rating={rating}, distance={distance}")
        features = np.array([[age, rating, distance]])
        features = features.reshape((features.shape[0], features.shape[1], 1))
        prediction = model.predict(features)
        predicted_time = prediction[0][0]
        st.subheader("Prediction")
        st.write(f"Predicted Delivery Time: {predicted_time:.2f} minutes")
        logger.info(f"Prediction: {predicted_time}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")
