import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import joblib
from data_collector import collect_data, series_to_supervised, read_dataset, get_filename
from datetime import datetime

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

# Download and save data
collect_data(from_symbol, to_symbol, exchange, datetime_interval)
current_datetime = datetime.now().date().isoformat()
data = read_dataset(get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime))
original_data = data
values = data.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Save scaler
scaler_filename = "model/scaler.save"
joblib.dump(scaler, scaler_filename)

num_features = len(data.columns)
num_past_days = 20
data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
values = data.values
num_obs = num_features * num_past_days

# Split into training and test set
n_train_days = int(0.9 * len(data))
train = values[:n_train_days, :]
test = values[n_train_days:, :]

train_X, train_y = train[:, :num_obs], train[:, -num_features]
test_X, test_y = test[:, :num_obs], test[:, -num_features]

train_X = train_X.reshape((train_X.shape[0], num_past_days, num_features))
test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))

# Setup the model
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# Train the model
history = model.fit(train_X, train_y, epochs=1000, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Plot the loss function
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_history.pdf')
plt.close()

# Predictions
yhat_test = model.predict(test_X)
yhat_train = model.predict(train_X)
test_X = test_X.reshape((test_X.shape[0], num_past_days * num_features))
train_X = train_X.reshape((train_X.shape[0], num_past_days * num_features))

# Invert scaling for forecast
inv_yhat_test = np.concatenate((yhat_test, test_X[:, -(num_features-1):]), axis=1)
inv_yhat_train = np.concatenate((yhat_train, train_X[:, -(num_features-1):]), axis=1)
inv_yhat_test = scaler.inverse_transform(inv_yhat_test)
inv_yhat_train = scaler.inverse_transform(inv_yhat_train)
inv_yhat_test = inv_yhat_test[:, 0]
inv_yhat_train = inv_yhat_train[:, 0]

# Plot result
plt.plot(original_data.index[num_past_days:], original_data['average'][num_past_days:], label='Target')
plt.plot(original_data.index[num_past_days:], list(inv_yhat_train) + list(inv_yhat_test), label='Predicted')
plt.axvline(x=original_data.index[n_train_days], label='Prediction Start', ymin=0.1, ymax=0.75, linestyle='--')
plt.legend()
plt.xlabel('Date')
plt.ylabel('BTC/USD')
plt.savefig('plots/performance_train_test.pdf')
plt.close()

plt.plot(original_data.index[n_train_days:], original_data['average'][n_train_days:], label='Target')
plt.plot(original_data.index[n_train_days + num_past_days:], list(inv_yhat_test), label='Predicted')
plt.legend()
plt.xlabel('Date')
plt.ylabel('BTC/USD')
plt.savefig('plots/performance_test.pdf')

# Save model
model.save('model/saved_model.model')
del model
