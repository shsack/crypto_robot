import requests
from datetime import datetime
import pandas as pd
import stockstats
import numpy as np


def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,\
        'datetime_interval should be one of %s' % supported_intervals

    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)

    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 2000, 'aggregate': 1,
              'e': exchange}
    request = requests.get(url, params=params)
    data = request.json()
    return data


def convert_to_dataframe(data):
    df = pd.io.json.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df


def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df


def read_dataset(filename):
    print('Reading data from %s' % filename)
    df = pd.read_csv(filename)
    df.datetime = pd.to_datetime(df.datetime)  # change type from object to datetime
    df = df.set_index('datetime')
    df = df.sort_index()  # sort by datetime
    return df


def collect_data(from_symbol, to_symbol, exchange, datetime_interval):

    data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
    df = convert_to_dataframe(data)
    df = df.rename(columns={'volumefrom': 'volume'})
    del df['volumeto']

    # Add features
    df = stockstats.StockDataFrame.retype(df)
    df.get("macd")
    df.get("rsi_14")
    df["average"] = (df["high"] + df["low"]) / 2

    # Filter empty data points and infinities
    df = filter_empty_datapoints(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    current_datetime = datetime.now().date().isoformat()
    filename = get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime)

    print('Saving data to %s' % filename)
    df.to_csv(filename, index=False)


from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

collect_data(from_symbol, to_symbol, exchange, datetime_interval)



