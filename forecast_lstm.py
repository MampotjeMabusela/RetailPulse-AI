import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("sales_data.csv")
data = df["sales"].values.reshape(-1, 1)

# Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, 30)

# Build model
model = Sequential([
    LSTM(64, input_shape=(30, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10)

# Predict
last_seq = data_scaled[-30:].reshape(1, 30, 1)
prediction = model.predict(last_seq)
print("Next day forecast:", scaler.inverse_transform(prediction))
