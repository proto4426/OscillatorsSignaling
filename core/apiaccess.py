from binance.client import Client
import numpy as np
import matplotlib.pyplot as plt
import configargparse

class ApiAccess:
    """
    Binance API Access Class
    """

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

    def __init__(self):
        self.args = self.parser.parse_known_args()[0]
        self.client = Client(self.args.key, self.args.secret)

        # Binance KLINE JSON response format example
        # [
        #         [
        #         1499040000000,      // Open time
        #         "0.01634790",       // Open
        #         "0.80000000",       // High
        #         "0.01575800",       // Low
        #         "0.01577100",       // Close
        #         "148976.11427815",  // Volume
        #         1499644799999,      // Close time
        #         "2434.19055334",    // Quote asset volume
        #         308,                // Number of trades
        #         "1756.87402397",    // Taker buy base asset volume
        #         "28.46694368",      // Taker buy quote asset volume
        #         "17928899.62484339" // Ignore
        #         ]
        # ]

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
        self.klines = self.client.get_historical_klines(self.args.left + self.args.right, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
        plt.plot(self.compute_MACD()[0][1200:1400])
        plt.ylabel('MACD')
        plt.show()
