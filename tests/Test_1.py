import yfinance as yf
import pandas as pd
def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker symbol between specified start and end dates.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the stock data with date as index.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data
print(get_stock_data('AAPL', '2020-01-01', '2020-12-31'))