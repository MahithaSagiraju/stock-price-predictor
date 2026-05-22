import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Download Data
data = yf.download('AAPL', start='2015-01-01', end='2026-01-01')

data_train = pd.DataFrame(data['Close'])

scaler = MinMaxScaler(feature_range=(0,1))

data_train_scale = scaler.fit_transform(data_train)

x_train = []
y_train = []

for i in range(100, data_train_scale.shape[0]):
    x_train.append(data_train_scale[i-100:i])
    y_train.append(data_train_scale[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build LSTM Model
model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True,
               input_shape=(x_train.shape[1],1)))

model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))

model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=32
)

model.save('stock_prediction_model.keras')

print("Model Saved Successfully")