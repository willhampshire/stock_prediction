import subprocess
from dotenv import load_dotenv
import os
import sys

from model.stock_data import fetch_stock_data, preprocess_data
#from model.model import create_model, create_dataset

load_dotenv()
cwd = os.getenv("CWD")
print(cwd)

# Start FastAPI server
fastapi_process = subprocess.Popen(["uvicorn", "fastapi_.main:app", "--reload"], cwd=cwd)

# Start Streamlit server
streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit/app.py"])

# Wait for both processes
fastapi_process.wait()
streamlit_process.wait()



### Create model

ticker_name = input("Please enter the ticker code for the company (e.g. NVDA): ")
try:
    ticker_name = str(ticker_name) # verify parse to string
except:
    print("Invalid.")
    sys.exit()

# Fetch and preprocess data
try:
    print("Fetching stock data")
    data = fetch_stock_data(ticker_name)
    data = preprocess_data(data)
except:
    print("Error - stock data")
    sys.exit()
