import yfinance as yf
from icecream import ic


def fetch_stock_data(ticker, period='2y', interval='1d'):
    """
    Fetches historical stock data from Yahoo Finance.

    :param ticker: Stock ticker symbol (e.g., 'NVDA').
    :param period: Data period (e.g., '5y' for 5 years).
    :param interval: Data interval (e.g., '1d' for daily).
    :return: DataFrame with stock data.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    return hist

def preprocess_data(data):
    """
    Preprocesses stock data by adding technical indicators.

    :param data: DataFrame with stock data.
    :return: Preprocessed DataFrame.
    """
    del data['Dividends']
    del data['Stock Splits']
    data['Return'] = data['Close'].pct_change()
    data['+1d'] = data['Close'].shift(-1)
    data['+7d'] = data['Close'].shift(-7)
    data = data.dropna()
    data.info()
    ic(data.head())
    return data

