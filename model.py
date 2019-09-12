import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collect_data import read_dataset
import joblib


def series_to_supervised(data, n_in=1, n_out=1):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)

    return agg

# exit()

#y = np.array(data['average']
#X = np.array(data

# lookback = 7
#
# test_size = int(.3 * len(data))
# X = []
# y = []
# for i in range(len(data) - lookback - 1):
#     t = []
#     for j in range(0, lookback):
#         t.append(data[[(i + j)], :])
#     X.append(t)
#     y.append(data[i + lookback, 1])
#
#
# print(X)
# print(y)
#
# exit()


data = read_dataset('BTC_USD_Bitstamp_day_2019-09-12.csv')
original_data = data
values = data.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Save scaler
scaler_filename = "scaler.save"
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

# print(len(train) + len(test))
# print(len(original_data.index[num_past_days:]))
# exit()

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
history = model.fit(train_X, train_y, epochs=100, batch_size=10, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# Plot the loss function
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.savefig('loss_history.pdf')
plt.close()

# Save model and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

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

# Invert scaling for target
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_y, test_X[:, -(num_features-1):]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

# Plot result
plt.plot(original_data.index[num_past_days:], original_data['average'][num_past_days:], label='Target')
plt.plot(original_data.index[num_past_days:], list(inv_yhat_train) + list(inv_yhat_test), label='Predicted')
plt.axvline(x=original_data.index[n_train_days], label='Prediction Start')
plt.legend()
plt.savefig('prediction.pdf')
