import subprocess
from dotenv import load_dotenv
import os
import sys

from tensorflow.keras.models import load_model


from model.stock_data import (
    fetch_stock_data,
    preprocess_data,
)  # model package contains model training functions
from model.model import build_train_model, analyse_distribute_results
from json_manage import json_state, data_file  # edit the fastapi state, data file


load_dotenv()
model_save_name = os.getenv("MODEL_SAVE_NAME")
ticker_name = os.getenv("TICKER")
time_period = int(os.getenv("TIME_PERIOD_LSTM"))


fastapi_process = subprocess.Popen(["uvicorn", "fastapi_main:app"])
streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit_app.py"])

json_state.set_state("training")


# Fetch and preprocess data
try:
    print("Fetching stock data")
    stock_data = fetch_stock_data(ticker_name)
    stock_data = preprocess_data(stock_data)
except:
    print("Error - stock data")
    sys.exit()


if os.path.exists(model_save_name):
    model = load_model(model_save_name)
    print("Loaded existing model")
else:
    build_train_model(stock_data, time_period=time_period, model_name=model_save_name)


results = data_file.read(mode=2)
# print(np.array(results).shape)
analyse_distribute_results(results)


json_state.set_state("trained")  # set state to trained


# pause until the processes terminate
fastapi_process.wait()
streamlit_process.wait()
