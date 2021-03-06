import numpy as np
import joblib
from keras.models import load_model
from data_collector import series_to_supervised, collect_data, read_dataset, get_filename
from datetime import datetime

loaded_model = load_model('model/saved_model.model')
scaler = joblib.load('model/scaler.save')

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

collect_data(from_symbol, to_symbol, exchange, datetime_interval)
current_datetime = datetime.now().date().isoformat()
orginal_data = read_dataset(get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime))

num_past_days = 20
num_features = len(orginal_data.columns)
num_obs = num_features * num_past_days
scaled = scaler.transform(orginal_data.values)
data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
values = data.values

# n_train_days = int(0.9 * len(data))

n_train_days = len(data) - 70

test = values[n_train_days:, :]
test_X = test[:, :num_obs]
test_X = test_X.reshape((test_X.shape[0], num_past_days, num_features))

yhat = loaded_model.predict(test_X)
tmp_X = test_X.reshape((test_X.shape[0], num_past_days * num_features))
inv_yhat_test = np.concatenate((yhat, tmp_X[:, -(num_features-1):]), axis=1)
inv_yhat_test = scaler.inverse_transform(inv_yhat_test)
inv_yhat_test = inv_yhat_test[:, 0]


# Backtest the model
prediced_changes = []
for i in range(len(inv_yhat_test) - 1):
    prediced_changes.append(inv_yhat_test[i+1] / inv_yhat_test[i])

open = orginal_data['open'][n_train_days + num_past_days + 1:]
close = orginal_data['close'][n_train_days + num_past_days + 1:]

model_returns = 1
for i, prediced_change in enumerate(prediced_changes):
    actual_change = close[i] / open[i]

    if prediced_change > 1 and actual_change > 1:
        tmp = actual_change
    if prediced_change > 1 and actual_change < 1:
        tmp = actual_change
    if prediced_change < 1 and actual_change < 1:
        tmp = 1 + (1 - actual_change)
    if prediced_change < 1 and actual_change > 1:
        tmp = 1 - (actual_change - 1)

    model_returns *= tmp

# Random buying and selling
random_returns_list = []
np.random.seed(10)

for _ in range(1000):
    random_returns = 1
    for o, c in zip(open, close):
        actual_change = c / o
        random_signal = np.random.randint(2)

        if random_signal == 1 and actual_change > 1:
            tmp = actual_change
        if random_signal == 1 and actual_change < 1:
            tmp = actual_change
        if random_signal == 0 and actual_change < 1:
            tmp = 1 + (1 - actual_change)
        if random_signal == 0 and actual_change > 1:
            tmp = 1 - (actual_change - 1)

        random_returns *= tmp
    random_returns_list.append(random_returns)


print('\n')
print('HODLing: {0:.2f} % of original investment.'.format(100 * (close[-1] / open[0])))
print('LSTM signal: {0:.2f} % of original investment.'.format(100 * model_returns))
print('Random signal: {0:.2f} % of original investment.'.format(100 * np.mean(np.array(random_returns_list))))




