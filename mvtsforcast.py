import keras
import numpy as np

lstm_model = keras.models.Sequential(
    [
        # Shape [batch, time, features] => [batch, time, lstm_units]
        keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        keras.layers.Dense(units=1),
    ]
)
