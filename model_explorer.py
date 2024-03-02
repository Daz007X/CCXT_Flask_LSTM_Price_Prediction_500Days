import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import joblib
import datetime
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from warnings import filterwarnings
import ccxt 

ETH_data = ccxt.binance().fetch_ohlcv('ETH/USDT', timeframe='1d')
df = pd.DataFrame(ETH_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df['Date'] = df['Date'].dt.date
df.set_index('Date', inplace=True)

def reshape_data(df, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(df)):
        X.append(df[i-time_steps:i,0])
        y.append(df[i,0])
    return np.array(X), np.array(y)

def predict_values_for_future_dates(model, data, start_date, num_dates, time_steps):
    predictions = []
    current_date = datetime.combine(start_date, datetime.min.time())
    for _ in range(num_dates):
        input_data = data[-time_steps:].values
        input_data = input_data.reshape(1, time_steps, 1)
        prediction = model.predict(input_data)
        predictions.append(prediction[0, 0])
        current_date += timedelta(days=1)
        data = pd.concat([data, pd.DataFrame({'close': prediction[0, 0]}, index=[current_date])])
    return predictions

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
time_steps = 20
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
train_data = dataset[:training_data_len, :]
test_data = dataset[training_data_len-time_steps:, :]
train_dates = data[:training_data_len].index
test_dates = data[training_data_len-time_steps:].index
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
X_train, y_train = reshape_data(train_data, time_steps)
X_test, y_test = reshape_data(test_data, time_steps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = Sequential([
    LSTM(40, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.20),
    LSTM(40, return_sequences=False),
    Dropout(0.20),
    Dense(16),
    Dropout(0.20),
    Dense(1)
])
model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics = ['mean_absolute_error']
             )
history = model.fit(
    X_train,
    y_train,
    callbacks=early_stop,
    epochs=50,
    batch_size=32,
    verbose=1
)
model.evaluate(X_test, y_test)
losses = pd.DataFrame(history.history)
y_pred_train = model.predict(X_train)
y_pred_train = scaler.inverse_transform(y_pred_train)
y_train_normal = y_train.reshape((y_train.shape[0], -1))
y_train_normal = scaler.inverse_transform(y_train_normal)
y_pred_test = model.predict(X_test)
y_pred_test = scaler.inverse_transform(y_pred_test)
y_test_normal = y_test.reshape((y_test.shape[0], -1))
y_test_normal = scaler.inverse_transform(y_test_normal)
last_results = pd.DataFrame({'close': data.values.reshape(-1, )}, index=data.index)
last_results['close'] = scaler.transform(last_results['close'].values.reshape(-1, 1))
start_date = data.index[-1]
num_dates = 500
p = predict_values_for_future_dates(model, last_results, start_date, num_dates+1, time_steps)
NEW_DATES = [data.index[-1]]
for _ in range(num_dates):
    data_append = datetime.date(data.index[-1] + pd.DateOffset(days=_+1))
    NEW_DATES.append(data_append)
RESULTS = pd.DataFrame({'close': p[:]}, index=NEW_DATES)
RESULTS['close'] = scaler.inverse_transform(RESULTS[['close']])
model.save("eth_lstm_model.h5") 
joblib.dump(scaler, 'eth_scaler.pkl')