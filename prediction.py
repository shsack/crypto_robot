import joblib
from data_collector import series_to_supervised, collect_data, read_dataset, get_filename
import numpy as np
from keras.models import load_model
from datetime import datetime
import requests
import urllib

# Load model and scalar
loaded_model = load_model('model/saved_model.model')
scaler = joblib.load('model/scaler.save')

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

collect_data(from_symbol, to_symbol, exchange, datetime_interval)
current_datetime = datetime.now().date().isoformat()
data = read_dataset(get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime))


num_past_days = 20
num_features = len(data.columns)
num_obs = num_features * num_past_days
scaled = scaler.transform(data.values)
data = series_to_supervised(scaled, n_in=num_past_days, n_out=1)
values = data.values

tmp = values[-2:, :]  #
tmp_X = tmp[:, :num_obs]
tmp_X = tmp_X.reshape((tmp_X.shape[0], num_past_days, num_features))

# loaded_model.compile()
yhat = loaded_model.predict(tmp_X)
tmp_X = tmp_X.reshape((tmp_X.shape[0], num_past_days * num_features))
inv_yhat = np.concatenate((yhat, tmp_X[:, -(num_features-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

change = 100 * ((inv_yhat[1] / inv_yhat[0]) - 1)

if inv_yhat[0] < inv_yhat[1]:
    ResultText = 'BTC/USD is predicted to increase by {0:.2f} % today.'.format(change)
    print(ResultText)
elif inv_yhat[0] > inv_yhat[1]:
    ResultText = 'BTC/USD is predicted to decrease by {0:.2f} % today.'.format(-change)
    print(ResultText)

# Send the prediction as a telegram message
ParsedRestultText = urllib.parse.quote_plus(ResultText)
requests.get("https://api.telegram.org/bot745226957:AAE_uWTumAKnSacuafxQWHVET_V-HKVIeGI/sendMessage?chat_id=-1001442571936&text={}".format(ParsedRestultText))

