import streamlit as st
import requests
from pandas import DataFrame as DF
from json_manage import data_file
from typing import Tuple

def plot_predictions():
    data: Tuple = data_file.read(mode=2)
    actual_df, predicted_df = DF(data[0]), DF(data[1])
    df = DF({'Actual': actual_df, 'Predicted': predicted_df})
    st.line_chart(df)

st.title('Stock Price Prediction')
st.write('Predictions for stock')

#features = st.text_input('Enter features as a list of numbers (e.g., [1.0, 2.0, 3.0]):')
#if st.button('Predict'):
#    try:
#        features = eval(features)
#        response = requests.post('http://localhost:8501/predict', json={'features': features})
#        predictions = response.json()['predictions']
#        st.write('Predictions:', predictions)
#    except Exception as e:
#        st.error(f"Error: {e}")
