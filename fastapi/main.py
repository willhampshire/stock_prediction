import sys
import numpy as np
import fastapi
from pydantic import BaseModel
import uvicorn

from sklearn.preprocessing import MinMaxScaler

from stock_data import fetch_stock_data, preprocess_data
from model import create_model, create_dataset


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

# Prepare data for model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]


time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = create_model((X_train.shape[1], 1))
model.fit(X_train, y_train, batch_size=1, epochs=1)





### FastAPI

app = fastapi.FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    return {'predictions': predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
