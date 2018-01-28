import sys
import configargparse
import pandas as pd

sys.path.insert(0, '../utils')

from DataSerializer import DataSerializer
from DataVisualizer import DataVisualizer
from constants import SINGLETON
from stockstats import StockDataFrame
from time import sleep


class MACDTest:

    # Argument parser
    parser = configargparse.get_argument_parser()

    args = None
    serializer = None
    visualizer = None
    # filename = None

    def execute(self):

        print("provisionning live data ...")
        while True:
            data = self.serializer.download_data(self.args.left, self.args.right, self.args.exchange, SINGLETON.DATETIME_INTERVAL_MINUTE, SINGLETON.LIVE_DATA_LIMIT)
            df = self.serializer.convert_to_dataframe(data)
            df = self.serializer.filter_empty_datapoints(df)

            # df = self.visualizer.read_dataset(self.filename)
            df = StockDataFrame.retype(df)
            df['macd'] = df.get('macd')
            # # print(df)
            # # print(df.columns)
            print(df)

            sleep(SINGLETON.REFRESH_DELAY)

    def __init__(self):
        # data = self.download_data(self.from_symbol, self.to_symbol, self.exchange, self.datetime_interval)
        # df = self.convert_to_dataframe(data)
        # df = self.filter_empty_datapoints(df)
        #
        # current_datetime = datetime.now().date().isoformat()
        # filename = self.get_filename(self.from_symbol, self.to_symbol, self.exchange, self.datetime_interval, current_datetime)
        # print('Saving data to %s' % filename)
        # df.to_csv(filename, index=False)
        # return filename
        self.args = self.parser.parse_known_args()[0]
        self.serializer = DataSerializer()
        self.visualizer = DataVisualizer()
        # self.filename = self.serializer.execute()
