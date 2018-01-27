import configargparse
from apiaccess import ApiAccess

def main():
    apiaccess = ApiAccess()
    apiaccess.execute()

def validate_usage(args):
    """
    Usage validation
    """
    if not args.exchange or not args.minutes or not args.left or not args.right:
        return False
    return True


if __name__ == "__main__":
    parser = configargparse.get_argument_parser()
    parser.add('--exchange', help='Target Exchange')
    parser.add('--minutes', help='MACD timespan in minutes')
    parser.add('--left', help='Left-hand side of the exchange pair')
    parser.add('--right', help='Right-hand side of the exchange pair')
    parser.add('-v', help='Verbose output', action='store_true')
    args = parser.parse_known_args()[0]

    if not validate_usage(args):
        print("Invalid or missing arguments. Usage : ")
        print("python main.py --exchange [Exchange name] --minutes [Number of minutes]--left [Currency] --right [Currency]")
        exit(-1)

    main()
