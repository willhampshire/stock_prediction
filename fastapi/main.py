import sys
from stock_data import fetch_stock_data, preprocess_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import fastapi
from pydantic import BaseModel
import uvicorn

# FastAPI
app = fastapi.FastAPI()

class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    return {'predictions': predictions.tolist()}


uvicorn.run(app, host='0.0.0.0', port=8000)

# Model training
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

ticker = input("Please enter the ticker code for the company (e.g. NVDA): ")
try:
    ticker = str(ticker) # verify parse to string
except:
    print("Invalid.")
    sys.exit()

# Fetch and preprocess data
try:
    data = fetch_stock_data(ticker)
    data = preprocess_data(data)
except:
    print("Error - stock data")
    sys.exit()

# Prepare data for model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

train_data = scaled_data[:int(len(scaled_data)*0.8)]
test_data = scaled_data[int(len(scaled_data)*0.8):]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = create_model((X_train.shape[1], 1))
model.fit(X_train, y_train, batch_size=1, epochs=1)
