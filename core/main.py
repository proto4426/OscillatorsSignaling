import sys
import configargparse

sys.path.insert(0, '../backtest')

from DataSerializer import DataSerializer
from DataVisualizer import DataVisualizer
from macd import MACDTest

def main(args):

    mode = args.mode

    if mode == "visual":
        launch_visualizaztion()
    elif mode == "backtest":
        launch_backtest()

def launch_backtest():
    test = MACDTest()
    test.execute()

def launch_visualizaztion():
    serializer = DataSerializer()
    visualizer = DataVisualizer()
    filename = serializer.execute()
    visualizer.visualize(filename, serializer.datetime_from, serializer.datetime_to, serializer.datetime_interval, serializer.from_symbol, serializer.to_symbol, serializer.exchange)

def validate_usage(args):
    """
    Usage validation
    """
    if not args.mode or not args.exchange or not args.minutes or not args.left or not args.right:
        return False
    return True


if __name__ == "__main__":
    parser = configargparse.get_argument_parser()
    parser.add('--mode', help='Execution mode')
    parser.add('--exchange', help='Target Exchange')
    parser.add('--minutes', help='MACD timespan in minutes')
    parser.add('--left', help='Left-hand side of the exchange pair')
    parser.add('--right', help='Right-hand side of the exchange pair')
    parser.add('-v', help='Verbose output', action='store_true')
    args = parser.parse_known_args()[0]

    if not validate_usage(args):
        print("Invalid or missing arguments. Usage : ")
        print("python main.py --mode [Execution Mode] --exchange [Exchange name] --minutes [Number of minutes]--left [Currency] --right [Currency]")
        exit(-1)

    main(args)
