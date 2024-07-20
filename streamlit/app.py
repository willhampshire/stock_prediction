import streamlit as st
import requests
import pandas as pd

def plot_predictions(actual, predicted):
    df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    st.line_chart(df)

st.title('Stock Price Prediction')
st.write('Predictions for NVDA stock')

features = st.text_input('Enter features as a list of numbers (e.g., [1.0, 2.0, 3.0]):')
if st.button('Predict'):
    try:
        features = eval(features)
        response = requests.post('http://fastapi:8000/predict', json={'features': features})
        predictions = response.json()['predictions']
        st.write('Predictions:', predictions)
    except Exception as e:
        st.error(f"Error: {e}")
