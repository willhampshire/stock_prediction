import yfinance as yf
from pandas import DataFrame as DF


def fetch_stock_data(ticker, period="2y", interval="1d"):
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


def preprocess_data(data) -> DF:
    """
    Preprocesses stock data by adding technical indicators.

    :param data: DataFrame with stock data.
    :return: Preprocessed DataFrame.
    """
    n = 7  # RSI window
    data["RSI"] = (
        data["Close"]
        .diff(1)
        .mask(data["Close"].diff(1) < 0, 0)
        .ewm(alpha=1 / n, adjust=False)
        .mean()
        .div(
            data["Close"]
            .diff(1)
            .mask(data["Close"].diff(1) > 0, -0.0)
            .abs()
            .ewm(alpha=1 / n, adjust=False)
            .mean()
        )
        .add(1)
        .rdiv(100)
        .rsub(100)
    )

    data["Volatility"] = data["High"] - data["Low"]
    data["Return"] = data["Close"].pct_change()
    data["+1d"] = data["Close"].shift(-1)
    data["+7d"] = data["Close"].shift(-7)

    data = data.drop(["Dividends", "Stock Splits", "High", "Low", "Open"], axis=1)
    data = data.dropna()
    data.info()
    return data
