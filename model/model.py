import os

import numpy as np
from pandas import DataFrame as DF
from typing import Tuple, List

from json_manage import data_file, metrics_file  # edit the fastapi state, data file

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input


def create_model(input_shape: Tuple[int, int, int]) -> Model:
    """
    Create the model with separate branches for +1d and +7d predictions.

    Parameters:
    - input_shape: tuple of shape Tuple[int, int, int], used to create input and LSTM layer

    Returns:
    - model: compiled model with 2 output nodes
    """
    input_shape = tuple(input_shape[1:])
    inputs = Input(shape=input_shape)

    # Shared LSTM layers
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50, return_sequences=False)(x)

    # Separate branches for +1d and +7d
    branch_1d = Dense(25, activation="relu")(x)
    output1 = Dense(1, activation="relu", name="output1")(branch_1d)  # For '+1d' target

    branch_7d = Dense(25, activation="relu")(x)
    output2 = Dense(1, activation="relu", name="output2")(branch_7d)  # For '+7d' target

    model = Model(inputs=inputs, outputs=[output1, output2])

    model.compile(
        optimizer="adam",
        loss={"output1": "mse", "output2": "mse"},
        metrics={"output1": "mse", "output2": "mse"},
    )

    return model


def create_sequences(data: np.ndarray, sequence_length: int = 10) -> np.ndarray:
    """
    Create sequences from the input data.

    Parameters:
    - data: np.ndarray, input data with shape (samples, features)
    - sequence_length: int, number of timesteps for each sequence

    Returns:
    - sequences: np.ndarray, reshaped data with shape (num_sequences, sequence_length, features)
    """
    num_samples = len(data)
    num_sequences = num_samples - sequence_length
    sequences = np.empty((num_sequences, sequence_length, data.shape[1]))

    for i in range(num_sequences):
        sequences[i] = data[i : i + sequence_length]

    return sequences


def build_train_model(data: DF, time_period: int, model_name: str) -> Model:
    """
    Trains the ML model, saves as .keras.

    :param data: data used to train the model
    :param time_period: days used to create sequences/windows for LSTM
    :param model_name: model save name
    :return: Model
    """

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = DF(scaler.fit_transform(data), columns=data.columns)

    # ic(scaled_data.head())

    predictors = list(data.columns)
    train = ["+1d", "+7d"]

    for col in train:
        predictors.remove(col)

    predictors_scaled_df = scaled_data[predictors]
    train_scaled_df = scaled_data[train]

    print(predictors_scaled_df.head(), train_scaled_df.head())

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(predictors_scaled_df),
        np.array(train_scaled_df),
        test_size=0.2,
        random_state=1,
    )

    x_train_sequences = create_sequences(x_train, time_period)
    y_train_sequences = create_sequences(y_train, time_period)
    x_test_sequences = create_sequences(x_test, time_period)
    y_test_sequences = create_sequences(y_test, time_period)

    print(x_train_sequences.shape, y_train_sequences.shape)

    input_shape = x_train_sequences.shape  # samples, timesteps, features

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model = create_model(input_shape)

    model_info = model.fit(
        x_train_sequences,
        [y_train_sequences[:, :, 0], y_train_sequences[:, :, 1]],
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        validation_data=(
            x_test_sequences,
            [y_test_sequences[:, :, 0], y_test_sequences[:, :, 1]],
        ),
        callbacks=[early_stopping],
    )

    model.save(model_name)

    y_test_result = model.predict(x_test_sequences)

    # `y_test_result` will be a list of two arrays, each containing the predictions for one of the outputs
    y_test_result_1d = y_test_result[0]  # Predictions for +1d
    y_test_result_7d = y_test_result[1]  # Predictions for +7d

    y_test_sequences_reduced = y_test_sequences[
        :, -1, :
    ]  # reduce the sequence dimension, take +1d and +7d at last index

    # Calculate accuracy metrics
    y_test_actual_1d = y_test_sequences_reduced[
        :, 0
    ]  # Actual values for +1d from the test set
    y_test_actual_7d = y_test_sequences_reduced[
        :, 1
    ]  # Actual values for +7d from the test set

    y_test_result_1d_flat = y_test_result_1d.flatten()
    y_test_result_7d_flat = y_test_result_7d.flatten()
    y_test_actual_1d_flat = y_test_actual_1d.flatten()
    y_test_actual_7d_flat = y_test_actual_7d.flatten()

    data_file.write(
        [y_test_actual_1d_flat, y_test_actual_7d_flat],
        [y_test_result_1d_flat, y_test_result_7d_flat],
    )

    check_data_json_df = DF(
        [
            y_test_actual_1d_flat,
            y_test_actual_7d_flat,
            y_test_result_1d_flat,
            y_test_result_7d_flat,
        ]
    )
    print(f"Written to data.json with NA: {check_data_json_df.isna().sum()}")
    print(check_data_json_df.head(10))
    return model


def analyse_distribute_results(results: Tuple[List[List], List[List]]):
    """
    Measures how good the prediction is using MSE, MAE, R2. Saves the results as JSON, so
    they can be displayed by the FastAPI service.

    :param results: results from .json
    :return: None
    """

    try:
        y_test_actual_1d = results[0][0]
        y_test_result_1d = results[1][0]
        y_test_actual_7d = results[0][1]
        y_test_result_7d = results[1][1]

        print(DF(results).head(10))
    except IndexError:
        print("data.json is probably empty")
        print(f"results.shape = {np.array(results).shape}")
        raise IndexError

    mse_1d = mean_squared_error(y_test_actual_1d, y_test_result_1d)
    mae_1d = mean_absolute_error(y_test_actual_1d, y_test_result_1d)
    r2_1d = r2_score(y_test_actual_1d, y_test_result_1d)

    mse_7d = mean_squared_error(y_test_actual_7d, y_test_result_7d)
    mae_7d = mean_absolute_error(y_test_actual_7d, y_test_result_7d)
    r2_7d = r2_score(y_test_actual_7d, y_test_result_7d)

    print(f"+1d - MSE: {mse_1d:.3e}, MAE: {mae_1d:.3e}, R2: {r2_1d:.3f}")
    print(f"+7d - MSE: {mse_7d:.3e}, MAE: {mae_7d:.3e}, R2: {r2_7d:.3f}")

    actual_stock_values = [y_test_actual_1d, y_test_actual_7d]
    predicted_stock_values = [y_test_result_1d, y_test_result_7d]

    data_file.write(actual_stock_values, predicted_stock_values)

    cwd = rf"{os.getcwd()}"
    fpath = cwd + r"/metrics.csv"
    np.savetxt(
        fpath, [[mse_1d, mae_1d, r2_1d], [mse_7d, mae_7d, r2_7d]], header="mse, mae, r2"
    )

    # save json for access via fastapi. CSV more convenient for testing and debug
    keys = ["MSE", "MAE", "R^2"]
    metrics_dict = {
        period: {metric_name: value for metric_name, value in zip(keys, metric_values)}
        for period, metric_values in {
            "1d": [mse_1d, mae_1d, r2_1d],
            "7d": [mse_7d, mae_7d, r2_7d],
        }.items()
    }
    metrics_file.write(metrics_dict)
