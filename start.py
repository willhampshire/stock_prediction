import subprocess
from dotenv import load_dotenv
import os
import sys

from model.stock_data import fetch_stock_data, preprocess_data
#from model.model import create_model, create_dataset
from json_manage import json_state # edit the fastapi json state variable


load_dotenv()
cwd = os.getenv("CWD")
print(cwd)

fastapi_process = subprocess.Popen(["uvicorn", "main:app", "--reload"], cwd=cwd)
streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit/app.py"])

json_state.set_state("training")

ticker_name = "NVDA"

# Fetch and preprocess data
try:
    print("Fetching stock data")
    data = fetch_stock_data(ticker_name)
    data = preprocess_data(data)
except:
    print("Error - stock data")
    sys.exit()




# pause until the processes terminate
fastapi_process.wait()
streamlit_process.wait()