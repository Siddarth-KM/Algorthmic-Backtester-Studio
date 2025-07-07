import yfinance as yf
 
print('Testing yfinance for AAPL (last 120 days)...')
df = yf.download('AAPL', period='120d')
print(df.tail())
print(f'Rows returned: {len(df)}') 