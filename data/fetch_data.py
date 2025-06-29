import yfinance as yf
import pandas as pd

def fetch_yfinance_data(tickers, start, end):
    """
    Download OHLCV data using yfinance.
    Returns a DataFrame indexed by date.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return data

