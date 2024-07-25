import subprocess
from dotenv import load_dotenv
import os
import sys

from model.stock_data import fetch_stock_data, preprocess_data # model package contains useful model training functions
from model.model import create_model, create_sequences
from json_manage import json_state, data_file # edit the fastapi state, data file
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame as DF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


load_dotenv()
cwd = os.getenv("CWD")
print(cwd)

fastapi_process = subprocess.Popen(["uvicorn", "main:app", "--reload"], cwd=cwd)
streamlit_process = subprocess.Popen(["streamlit", "run", "streamlit/app.py"])

json_state.set_state("training")

ticker_name = "NVDA"
time_period = 31 # days used to create sliding window

# Fetch and preprocess data
try:
    print("Fetching stock data")
    data = fetch_stock_data(ticker_name)
    data = preprocess_data(data)
except:
    print("Error - stock data")
    sys.exit()

#data.plot.line(y='Close', use_index=True)
#plt.show()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = DF(scaler.fit_transform(data), columns=data.columns)

#ic(scaled_data.head())

predictors = list(data.columns)
train = ["+1d","+7d"]

for col in train:
    predictors.remove(col)


predictors_scaled_df = scaled_data[predictors]
train_scaled_df = scaled_data[train]

print(predictors_scaled_df.head(), train_scaled_df.head())


x_train, x_test, y_train, y_test = train_test_split(np.array(predictors_scaled_df), np.array(train_scaled_df),
                                                    test_size=0.2, random_state=1)


x_train_sequences = create_sequences(x_train, time_period)
y_train_sequences = create_sequences(y_train, time_period)
x_test_sequences = create_sequences(x_test, time_period)
y_test_sequences = create_sequences(y_test, time_period)

print(x_train_sequences.shape, y_train_sequences.shape)

input_shape = x_train_sequences.shape # samples, timesteps, features

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model = create_model(input_shape)

model_info = model.fit(x_train_sequences, [y_train_sequences[:, :, 0], y_train_sequences[:, :, 1]],
                    epochs=25,
                    batch_size=32,
                    validation_split=0.2,
                    validation_data=(x_test_sequences, [y_test_sequences[:, :, 0], y_test_sequences[:, :, 1]]),
                    callbacks=[early_stopping])

model.save("stock_1,7day_model.keras")


y_test_result = model.predict(x_test_sequences)

# `y_test_result` will be a list of two arrays, each containing the predictions for one of the outputs
y_test_result_1d = y_test_result[0]  # Predictions for +1d
y_test_result_7d = y_test_result[1]  # Predictions for +7d

y_test_sequences_reduced = y_test_sequences[:, -1, :] # reduce the sequence dimension, take +1d and +7d at last index

# Calculate accuracy metrics
y_test_actual_1d = y_test_sequences_reduced[:, 0]  # Actual values for +1d from the test set
y_test_actual_7d = y_test_sequences_reduced[:, 1]  # Actual values for +7d from the test set

mse_1d = mean_squared_error(y_test_actual_1d, y_test_result_1d)
mae_1d = mean_absolute_error(y_test_actual_1d, y_test_result_1d)
r2_1d = r2_score(y_test_actual_1d, y_test_result_1d)

mse_7d = mean_squared_error(y_test_actual_7d, y_test_result_7d)
mae_7d = mean_absolute_error(y_test_actual_7d, y_test_result_7d)
r2_7d = r2_score(y_test_actual_7d, y_test_result_7d)

print(f'+1d - MSE: {mse_1d:.3e}, MAE: {mae_1d:.3e}, R2: {r2_1d:.3f}')
print(f'+7d - MSE: {mse_7d:.3e}, MAE: {mae_7d:.3e}, R2: {r2_7d:.3f}')




y_test_result_1d_flat = y_test_result_1d.flatten()
y_test_result_7d_flat = y_test_result_7d.flatten()
y_test_actual_1d_flat = y_test_actual_1d.flatten()
y_test_actual_7d_flat = y_test_actual_7d.flatten()


actual_stock_values = [y_test_actual_1d_flat, y_test_actual_7d_flat]
predicted_stock_values = [y_test_result_1d_flat, y_test_result_7d_flat]

data_file.write(actual_stock_values, predicted_stock_values)


json_state.set_state("trained")


# pause until the processes terminate
fastapi_process.wait()
streamlit_process.wait()