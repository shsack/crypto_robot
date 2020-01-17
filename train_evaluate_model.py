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
from sklearn.model_selection import train_test_split

def one_hot_vector_encoding(data):
    # threshold = 0.015
    y = []
    for i, change in enumerate(data.values):
        # if change < - threshold:
        #     y.append([1, 0, 0, 0])
        # elif - threshold <= change < 0:
        #     y.append([0, 1, 0, 0])
        # elif 0 <= change < threshold:
        #     y.append([0, 0, 1, 0])
        # elif change >= threshold:
        #     y.append([0, 0, 0, 1])
        # else:
        #     print('Error, check entries!')
        if change < 0:
            y.append([1, 0])
        elif change >= 0:
            y.append([0, 1])
    return y

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

# Download and save data
collect_data(from_symbol, to_symbol, exchange, datetime_interval)
current_datetime = datetime.now().date().isoformat()
data = read_dataset(get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime))

data["change"] = data["average"].pct_change()
data.dropna(inplace=True)

num_past_days = 10

y = data["change"]
y = one_hot_vector_encoding(y)[num_past_days:]

# original_data = data
# values = data.values

# data = data.drop(columns="change")
# data = one_hot_vector_encoding(data)
values = data.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# exit()
# scaled = values
# Save scaler
# scaler_filename = "model/scaler.save"
# joblib.dump(scaler, scaler_filename)

num_features = len(data.columns)
# num_past_days = 10

# y = data["average"].pct_change()
# y.dropna(inplace=True)

# y = one_hot_vector_encoding(y)[num_past_days-1:]


data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
X = data.values


num_obs = num_features * num_past_days

# Split into training and test set
# n_train_days = int(0.8 * len(data))
# train = values[:n_train_days, :]  # ??
# test = values[n_train_days:, :]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# print(num_features)
# print(num_obs)
#
# print(train[0, -num_features])
# exit()


X_train = X_train[:, :num_obs]
X_test = X_test[:, :num_obs]
y_train = np.array(y_train)
y_test = np.array(y_test)


# train_X, train_y = train[:, :num_obs], train[:, -1]  # prediction = change
# test_X, test_y = test[:, :num_obs], test[:, -1]


# train_y = np.array(y[:n_train_days])
# test_y = np.array(y[n_train_days:-1])

# X_train = X_train.reshape((1, X_train.shape[0], X_train.shape[1]))  # TODO: understand this
# X_test = X_test.reshape((1, X_test.shape[0], X_test.shape[1]))
#
# print(X_train.shape)
# print(X_test.shape)
#
# exit()
#
X_train = X_train.reshape((X_train.shape[0], num_past_days, num_features))  # TODO: understand this
X_test = X_test.reshape((X_test.shape[0], num_past_days, num_features))


# Setup the model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(Dense(1))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=5, validation_data=(X_test, y_test), verbose=2, shuffle=False)


exit()
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
