import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Electricity Bill Predictor",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Electricity Bill Prediction")
st.write("Predict your monthly electricity bill using a neural network.")

MODEL_FILE = "electricity_model.keras"
SCALER_FILE = "scaler.pkl"

# Check whether the model has been trained
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    st.warning("Trained model not found.")
    st.info("First run the training program:")
    st.code("python train_model.py")
    st.stop()

# Load model and scaler
@st.cache_resource
def load_artifacts():
    trained_model = load_model(MODEL_FILE)
    with open(SCALER_FILE, "rb") as file:
        trained_scaler = pickle.load(file)
    return trained_model, trained_scaler

model, scaler = load_artifacts()

st.subheader("Enter Your Details")

units = st.number_input(
    "Monthly electricity consumption (units / kWh)",
    min_value=0,
    max_value=2000,
    value=200,
    step=10
)

occupants = st.number_input(
    "Number of people in the household",
    min_value=1,
    max_value=20,
    value=4,
    step=1
)

temperature = st.number_input(
    "Average daily temperature (°C)",
    min_value=-10,
    max_value=50,
    value=32,
    step=1
)

if st.button("Predict Electricity Bill", type="primary",
             use_container_width=True):

    input_data = pd.DataFrame(
        [[units, occupants, temperature]],
        columns=["Units", "Occupants", "Temperature"]
    )

    scaled_data = scaler.transform(input_data)
    prediction = float(
        model.predict(scaled_data, verbose=0)[0][0]
    )

    if np.isfinite(prediction):
        # A bill cannot be negative
        prediction = max(0, prediction)

        st.subheader("Prediction Result")
        st.metric(
            label="Estimated Monthly Electricity Bill",
            value=f"₹ {prediction:,.2f}"
        )

        st.caption(
            "This is an educational estimate based on sample data. "
            "Actual bills depend on your tariff, taxes, fixed charges, "
            "and other billing rules."
        )
    else:
        st.error("Unable to calculate the bill. Please try again.")

st.divider()
st.caption("Mini Project | Neural Networks | TensorFlow + Streamlit")