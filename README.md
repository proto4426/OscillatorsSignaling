# Cryptotrading Signaling via Oscillators
MACD oscillator signaling based on API data in Python

<p align="center">
  <img src="http://www.reactiongifs.us/wp-content/uploads/2013/10/nuh_uh_conan_obrien.gif" alt="Oscillator lol j/k"/>
</p>

## Dependencies :
The following command should be executed prior to the first execution :

```
pip install -r requirements.txt
```

## Usage :

python main.py --exchange [Exchange name] --minutes [Number of minutes]--left [Currency] --right [Currency]

Where the parameters are the following :
- `exchange` : Target exchange

- `minutes` : MACD timespan in minutes

- `left` : Left-hand side of the exchange pair

- `right` :  Right-hand side of the exchange pair

## Acknowledgements
This program is built upon [CryptoCompare](https://www.cryptocompare.com/) API. Data visualization is provided based on the work of [Roman Orac](https://romanorac.github.io/).
