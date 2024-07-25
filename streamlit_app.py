import streamlit as st
import numpy as np
import pandas as pd
from json_manage import data_file
from typing import Tuple


def plot_predictions():
    # Read the data
    data: Tuple = data_file.read(mode=2)

    # Extract real and predicted values
    actual = np.array(data[0][0])  # Assuming data[0] is an array of real values
    predicted = np.array(data[1][0])  # Assuming data[1] is an array of predicted values

    actual_flat = actual
    predicted_flat = predicted

    # Create DataFrames for plotting
    df = pd.DataFrame({
        'Actual': actual_flat,
        'Predicted': predicted_flat
    })

    # Use Streamlit to plot the data
    st.line_chart(df)


# Example Streamlit app
st.title('Stock Price Prediction')
st.write('Predictions for stock')

plot_predictions()