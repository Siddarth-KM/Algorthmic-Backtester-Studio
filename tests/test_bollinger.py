import sys
import traceback
from backtester_main import Bollinger_Band
import yfinance as yf
import datetime as dt

try:
    # Test Bollinger Band strategy directly
    print("Testing Bollinger Band strategy...")
    
    # Download some test data
    end = dt.datetime.now()
    start = end - dt.timedelta(days=60)  # 2 months of data
    df = yf.download("AAPL", start=start, end=end)
    
    print(f"Downloaded {len(df)} rows of data")
    print("Data columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    
    # Test the strategy
    result = Bollinger_Band(df, None)
    print("Strategy completed successfully!")
    print("Result:", result)
    
except Exception as e:
    print("Error occurred:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc() 