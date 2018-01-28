from binance.client import Client
import numpy as np
import configargparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from stockstats import StockDataFrame
from math import pi
from bokeh.plotting import figure, show, output_notebook, output_file
output_notebook()

class DataSerializer:
    """
    Serializer for exchange API data
    """

    from_symbol = None
    to_symbol = None
    exchange = None
    datetime_interval = 'minute'

    current_time_full = datetime.now().time().isoformat()
    tokens = current_time_full.split(":",2)
    print(tokens[0] + ":" + tokens[1])

    delta_time_full = (datetime.now() - timedelta(minutes=15)).time().isoformat()
    tokens = delta_time_full.split(":",2)
    print(tokens[0] + ":" + tokens[1])

    datetime_from = None
    datetime_to = None
    # Default interval count
    TOTAL_PERIODS = 1440

    # MACD time frames
    MACD_HI = 26
    MACD_MID = 12
    MACD_SIGNAL = 9

    # Argument parser
    parser = configargparse.get_argument_parser()
    # Binance API client object
    client = None
    # Binance data
    klines = None

    def get_filename(self, from_symbol, to_symbol, exchange, datetime_interval, download_date):
        return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


    def download_data(self, from_symbol, to_symbol, exchange, datetime_interval):
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

    def convert_to_dataframe(self, data):
        df = pd.io.json.json_normalize(data, ['Data'])
        df['datetime'] = pd.to_datetime(df.time, unit='s')
        df = df[['datetime', 'low', 'high', 'open',
                 'close', 'volumefrom', 'volumeto']]
        return df


    def filter_empty_datapoints(self, df):
        indices = df[df.sum(axis=1) == 0].index
        print('Filtering %d empty datapoints' % indices.shape[0])
        df = df.drop(indices)
        return df

    def __init__(self):
        self.args = self.parser.parse_known_args()[0]
        # self.client = Client(self.args.key, self.args.secret)
        self.exchange = self.args.exchange
        self.from_symbol = self.args.left
        self.to_symbol = self.args.right

        current_time_full = datetime.now().time().isoformat()
        current_tokens = current_time_full.split(":",2)

        delta_time_full = (datetime.now() - timedelta(minutes=int(self.args.minutes))).time().isoformat()
        delta_date_full = (datetime.now() - timedelta(minutes=int(self.args.minutes))).date().isoformat()
        delta_tokens = delta_time_full.split(":",2)

        self.datetime_from = delta_date_full + " " + delta_tokens[0] + ":" + delta_tokens[1]
        self.datetime_to = datetime.now().date().isoformat() + " " + current_tokens[0] + ":" + current_tokens[1]

    def compute_SMA(self, interval, start_epoch, points):
        """
        Computes Simple Moving Average based on API data fetched against interval on specified time
        self.klines must not be empty
        """

        return (sum((list(map(lambda tick: float(tick[4]), self.klines)))[start_epoch - interval:start_epoch])) / interval

    def compute_EMA(self, interval, start_epoch, points):
        """
        Computes Exponential Moving Average based on API data fetched against interval on specified time
        """
        sma = self.compute_SMA(interval, start_epoch, self.klines)
        multiplier = (2 / (interval + 1))
        closings = (list(map(lambda tick: float(tick[4]), self.klines)))[start_epoch - interval:start_epoch]
        prev_ema = sma
        emas = [None] * (interval)

        for index, closing in enumerate(closings):
            ema = (closing - prev_ema) * multiplier + prev_ema
            emas[index] = ema
            prev_ema = ema

        return emas[-1]

    def compute_MACD(self):
        """
        Computes MA Convergence/Divergence plot points based on EMAs 26/12/9
        """

        macd_line = [None] * self.TOTAL_PERIODS
        for i in range(0 + self.MACD_HI, self.TOTAL_PERIODS):
            point = (self.compute_EMA(self.MACD_MID, i, self.klines) - self.compute_EMA(self.MACD_HI, i, self.klines))
            macd_line[i] = point

        signal_line = [None] * self.TOTAL_PERIODS
        for i in range(0 + self.MACD_SIGNAL, len(signal_line)):
            point = self.compute_EMA(self.MACD_SIGNAL, i, macd_line)
            signal_line[i] = point
        histogram = list(map(lambda pair: pair[0] - pair[1], list(zip(macd_line[self.MACD_HI:len(macd_line)], signal_line[self.MACD_SIGNAL:len(signal_line)]))))
        return (macd_line, signal_line, histogram)

    def execute(self):

        data = self.download_data(self.from_symbol, self.to_symbol, self.exchange, self.datetime_interval)
        df = self.convert_to_dataframe(data)
        df = self.filter_empty_datapoints(df)

        current_datetime = datetime.now().date().isoformat()
        filename = self.get_filename(self.from_symbol, self.to_symbol, self.exchange, self.datetime_interval, current_datetime)
        print('Saving data to %s' % filename)
        df.to_csv(filename, index=False)
        return filename
