import streamlit as st
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page title
st.title("Food Delivery Time Prediction")

# Load model with error handling
try:
    logger.info("Attempting to load model: food_delivery_model.keras")
    model = tf.keras.models.load_model('food_delivery_model.keras')  # Add custom_objects if needed
    st.success("Model loaded successfully")
    logger.info("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    logger.error(f"Model loading failed: {str(e)}")
    st.stop()

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Details")
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, step=0.1)
    distance = st.number_input("Distance (meters)", min_value=0, step=100)
    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    try:
        logger.info(f"Input features: age={age}, rating={rating}, distance={distance}")
        inputs = np.array([[age, rating, distance]])
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], 1))  # LSTM shape
        logger.info(f"Reshaped inputs: {inputs.shape}")
        prediction = model.predict(inputs, verbose=0)
        logger.info(f"Prediction output: {prediction}")
        time = prediction[0][0]
        st.subheader("Prediction")
        st.success(f"Predicted Delivery Time: {time:.2f} minutes")
        logger.info(f"Predicted time: {time:.2f} minutes")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        logger.error(f"Prediction failed: {str(e)}")
