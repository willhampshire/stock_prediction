import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from typing import Tuple


def create_model(input_shape :Tuple[int, int, int]) -> Model:
    """
        Create the model.

        Parameters:
        - input_shape: tuple of shape Tuple[int, int, int], used to create input and LSTM layer

        Returns:
        - model: compiled model with 2 output nodes
        """
    input_shape = tuple(input_shape[1:])
    inputs = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True)(inputs)
    x = LSTM(50, return_sequences=False)(x)
    x = Dense(25, activation='relu')(x)
    output1 = Dense(1, activation='relu', name='output1')(x)  # For '+1d' target
    output2 = Dense(1, activation='relu', name='output2')(x)  # For '+7d' target
    model = Model(inputs=inputs, outputs=[output1, output2])

    model.compile(optimizer='adam',
                  loss={'output1': 'mse', 'output2': 'mse'},
                  metrics={'output1': 'mse', 'output2': 'mse'})

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
        sequences[i] = data[i:i + sequence_length]

    return sequences



