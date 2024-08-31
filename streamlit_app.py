import streamlit as st
import numpy as np
import pandas as pd
from json_manage import data_file
from typing import Tuple
from pandas import DataFrame as DF
import time


def plot_predictions():
    # Read the data
    data: Tuple = data_file.read(mode=2)
    time.sleep(0.5)

    # Extract real and predicted values
    actual = np.array(data[0])  # Assuming data[0] is an array of real values
    predicted = np.array(data[1])  # Assuming data[1] is an array of predicted values

    actual_1d = actual[0]
    predicted_1d = predicted[0]
    actual_7d = actual[1]
    predicted_7d = predicted[1]

    # Create DataFrames for plotting
    df_1d = pd.DataFrame({"Actual": actual_1d, "Predicted": predicted_1d})
    df_7d = pd.DataFrame({"Actual": actual_7d, "Predicted": predicted_7d})

    df_1d["Predicted"] = df_1d["Predicted"].shift(1)
    df_7d["Predicted"] = df_7d["Predicted"].shift(7)

    df_1d["Pred. Diff"] = df_1d["Predicted"] - df_1d["Actual"]
    df_7d["Pred. Diff"] = df_7d["Predicted"] - df_7d["Actual"]

    print(df_1d.head(10))
    print(df_7d.head(10))

    st.subheader("1-Day Ahead Prediction")
    st.line_chart(df_1d)
    st.subheader("7-Day Ahead Prediction")
    st.line_chart(df_7d)


st.title("Stock Price Prediction")

plot_predictions()
