# type: ignore
import pandas as pd
import yfinance as yf
import datetime as dt
from matplotlib import pyplot as plt
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from xgboost import XGBClassifier
from sklearn.utils import class_weight, resample
import numpy as np
from scipy.stats import linregress
import warnings
from typing import Optional, List, Any, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Import the plot function from backtester_main
from backtester_main import plot_trades_ml

# --- BEGIN UNWRAPPED SCRIPT ---
def run_ml_backtest(ticker=None, download_start_date=None, test_period=None, buy_confidence=0.5, sell_confidence=0.9, df_filtered=None):
    """
    Run ML backtest with configurable parameters
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'BTC-USD')
        download_start_date (str): Download start date in format 'YYYY-MM-DD' or choice '1', '2', '3', '4', '5'
        test_period (str): Test period choice '1' (3 months), '2' (6 months), '3' (YTD), '4' (1 year), '5' (2 years)
        buy_confidence (float): Confidence threshold for buy signals (default 0.5)
        sell_confidence (float): Confidence threshold for sell signals (default 0.9)
        df_filtered (pd.DataFrame): Optional filtered DataFrame for plotting (user-requested date range only)
    """
    # Handle interactive mode if no parameters provided
    if ticker is None:
        return {"error": "Ticker is required"}
    else:
        stock = ticker.upper()
    
    if download_start_date is None:
        start_choice = '3'  # Default to 2010
    else:
        start_choice = download_start_date
    
    if test_period is None:
        test_choice = '4'  # Default to 1 year
    else:
        test_choice = test_period
    
    end = dt.datetime.now()
    # Determine start_date based on choice
    START_DATE_MAP = {
        '1': '2000-01-01',
        '2': '2005-01-01',
        '3': '2010-01-01',
        '4': '2015-01-01',
        '5': '2010-01-01'
    }
    start_date = START_DATE_MAP.get(start_choice, '2010-01-01')
    # Download data with error handling
    df: Optional[pd.DataFrame] = yf.download(stock, start=start_date, end=end.strftime('%Y-%m-%d'), auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"Failed to download data for {stock}. Please check the ticker symbol and try again.")
        return {"error": f"Failed to download data for {stock}"}
    
    # Ensure df is a proper DataFrame with DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"Warning: DataFrame index is not DatetimeIndex, converting...")
        df.index = pd.to_datetime(df.index)

    df['close_series'] = df['Close']
    df['high_series'] = df['High']
    df['low_series'] = df['Low']
    df['volume_series'] = df['Volume']

    today = pd.Timestamp(end)
    def get_cutoff_date(choice):
        if choice == '1':
            return (today - pd.DateOffset(months=3)).strftime('%Y-%m-%d')
        elif choice == '2':
            return (today - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
        elif choice == '3':
            return pd.Timestamp(today.year, 1, 1).strftime('%Y-%m-%d')
        elif choice == '4':
            return (today - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        elif choice == '5':
            return (today - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        elif choice == '6':
            return (today - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        else:
            return (today - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    cutoff_date = get_cutoff_date(test_choice)

    def calc_slope(series: pd.Series) -> float:
        if series.isna().any():
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return float(slope)

    df['slope_price_10d'] = df['close_series'].rolling(window=10).apply(calc_slope, raw=False)
    df['ATR'] = AverageTrueRange(high=df['high_series'], low=df['low_series'], close=df['close_series']).average_true_range()
    df['SMA_20'] = df['close_series'].rolling(window=20).mean().shift(1)
    df['SD'] = df['close_series'].rolling(window=20).std().shift(1)
    df['Upper'] = df['SMA_20'] + 2*df['SD']
    df['Lower'] = df['SMA_20'] - 2*df['SD']
    df['width'] = (df['Upper'] - df['Lower']) / df['SMA_20']
    df['percent_b'] = (df['close_series'] - df['Lower']) / (df['Upper'] - df['Lower'])
    rolling_mean = df['close_series'].rolling(window=10).mean()
    df['rolling_std_10'] = df['close_series'].rolling(window=10).std()
    df['z_score_close'] = (df['close_series'] - rolling_mean) / df['rolling_std_10']
    df["Ema_10"] = df['Close'].ewm(span=10, adjust=False).mean()
    df['volume_pct_change'] = df['Volume'].pct_change()
    df['RSI'] = RSIIndicator(close=df['close_series']).rsi()
    df['roc_10'] = ROCIndicator(close=df['close_series']).roc()
    df['momentum'] = df['close_series'] - df['close_series'].shift(10)
    df['ema_diff'] = df['Ema_10'] - df['SMA_20']
    df['price_position'] = (df['close_series'] - df['Lower']) / (df['Upper'] - df['Lower'])
    df['past_10d_return'] = df['close_series'] / df['close_series'].shift(10) - 1
    df['rsi_14_diff'] = df['RSI'] - df['RSI'].shift(5)
    df['price_vs_ema'] = df['close_series'] - df['Ema_10']
    df['adx_14'] = ADXIndicator(high=df['high_series'], low=df['low_series'], close=df['close_series']).adx()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['close_series'], volume=df['volume_series']).on_balance_volume()
    df['log_return'] = np.log(df['close_series'] / df['close_series'].shift(1))
    df['volatility'] = df['log_return'].rolling(10).std()
    df['ema_ratio'] = df['Ema_10'] / df['SMA_20']
    for lag in range(1, 6):
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df_train: pd.DataFrame = df[df.index < cutoff_date]
    df_test: pd.DataFrame = df[df.index >= cutoff_date]
    indicators = ['width', 'percent_b', 'z_score_close', 'rolling_std_10', 'slope_price_10d','log_return', 'ema_ratio', 'RSI' ,'SMA_20', 'Ema_10', 'roc_10','Upper', 'Lower', 'ATR', 'volume_pct_change', 'ema_diff' , 'price_position', 'past_10d_return','rsi_14_diff','momentum','price_vs_ema', 'volatility','adx_14', 'OBV', 'close_lag_1'  ,'close_lag_2' ,'close_lag_3' ,'close_lag_4' ,'close_lag_5']
    df_train = df_train.copy()
    df_train['future_return'] = df_train['close_series'].shift(-10) / df_train['close_series'] - 1
    df_train['Label'] = 1
    df_train.loc[df_train['future_return'] > 0.015, 'Label'] = 2
    df_train.loc[df_train['future_return'] < -0.025, 'Label'] = 0
    df_test = df_test.copy()
    df_test.loc[:, 'future_return'] = df_test['close_series'].shift(-10) / df_test['close_series'] - 1
    df_test.loc[:, 'Label'] = 1
    df_test.loc[df_test['future_return'] > 0.015, 'Label'] = 2
    df_test.loc[df_test['future_return'] < -0.025, 'Label'] = 0
    df_train_ml: pd.DataFrame = df_train[indicators + ['Label']].dropna()
    df_0: pd.DataFrame = df_train_ml[df_train_ml['Label'] == 0]
    df_1: pd.DataFrame = df_train_ml[df_train_ml['Label'] == 1]
    df_2: pd.DataFrame = df_train_ml[df_train_ml['Label'] == 2]
    # Check if any class is missing
    if len(df_0) == 0 or len(df_1) == 0 or len(df_2) == 0:
        print(f"One or more classes missing in training data. Class counts: 0={len(df_0)}, 1={len(df_1)}, 2={len(df_2)}")
        return
    minority_size = min(len(df_0), len(df_1), len(df_2))
    if minority_size > 0:
        df_0 = resample(df_0, replace=False, n_samples=minority_size, random_state=42)
        df_1 = resample(df_1, replace=False, n_samples=minority_size, random_state=42)
        df_2 = resample(df_2, replace=False, n_samples=minority_size, random_state=42)
        df_balanced: pd.DataFrame = pd.concat([df_0, df_1, df_2]).sample(frac=1, random_state=42)
        df_balanced = df_balanced.sample(frac=1, random_state=42)
    else:
        df_balanced: pd.DataFrame = pd.concat([df_0, df_1, df_2])
    x_train: pd.DataFrame = df_balanced[indicators]
    y_train: pd.Series = df_balanced['Label']
    x_test: pd.DataFrame = df_test[indicators].dropna()
    y_test: pd.Series = df_test.loc[x_test.index, 'Label']
    df_train = df[df.index < cutoff_date]
    df_test = df[df.index >= cutoff_date]
    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    sample_weights = class_weight.compute_sample_weight('balanced', y_train)
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(x_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(x_test)
    y_pred_shifted = np.roll(y_pred, 1)
    y_pred_shifted[0] = y_pred_shifted[1]
    probs = model.predict_proba(x_test)
    importances = model.feature_importances_
    feat_names = x_train.columns
    importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    importance_df.sort_values(by='Importance', ascending=True, inplace=True)
    in_position = False
    entry_price = 0
    returns: List[float] = []
    entry_dates: List[Any] = []
    exit_dates: List[Any] = []
    df_test = df.loc[x_test.index].copy()
    df_test['confidence'] = probs[:, 2]
    i = 0
    probs_all = model.predict_proba(x_test)
    hold = -1
    while i < len(df_test) - 1:
        probs_row = probs_all[i]
        predicted_class = np.argmax(probs_row)
        confidence = np.max(probs_row)
        price_today = df_test['close_series'].iloc[i]
        tomorrow_price = df_test['close_series'].iloc[i + 1]
        if not in_position and predicted_class == 2 and confidence >= buy_confidence:
            in_position = True
            entry_price = tomorrow_price
            entry_dates.append(df_test.index[i + 1])
            hold = -1
            stop = entry_price - 1.5*df_test['ATR'].iloc[i]
        if in_position and hold >= 10 and ((predicted_class == 0 and confidence >= sell_confidence) or price_today <= stop):
            exit_price = tomorrow_price
            trade_return = (exit_price/ entry_price) - 1
            returns.append(trade_return)
            exit_dates.append(df_test.index[i + 1])
            in_position = False
        elif in_position and df_test['close_series'].iloc[i] > df_test['close_series'].iloc[i-1]:
            stop = df_test['close_series'].iloc[i] - 1.5*df_test['ATR'].iloc[i]
        if in_position:
            hold += 1
        i += 1
    if in_position:
        exit_price = df_test['close_series'].iloc[-1]
        trade_return = (exit_price/ entry_price) - 1
        returns.append(trade_return)
        exit_dates.append(df_test.index[-1])
        in_position = False
    df_test_true = df.iloc[-len(x_test):]
    
    # Check if df_test_true is empty or has issues
    if df_test_true.empty:
        return {"error": "No test data available for performance calculation"}
    
    if len(df_test_true) < 2:
        return {"error": "Insufficient test data for performance calculation"}
    

    
    firstclose = df_test_true['close_series'].iloc[0]
    finalclose = df_test_true['close_series'].iloc[-1]  # Changed from -2 to -1
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    final_value = 100 * np.prod([1 + r for r in returns])
    calcvalue = final_value - 100
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    Trades = len(returns)
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 and not np.isnan(np.std(returns)) else None
    
    # Calculate new advanced stats
    if returns:
        # 3. Calmar Ratio (Annual Return / Max Drawdown)
        # Calculate cumulative returns for drawdown
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Annualize the return (assuming daily data, multiply by ~252 trading days)
        days_in_test = (df_test.index[-1] - df_test.index[0]).days
        annual_return = (final_value / 100 - 1) * (365 / days_in_test) if days_in_test > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else None
        
        # 4. Sortino Ratio (Return / Downside Deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else None
        
        # 8. Profit Factor (Gross Profit / Gross Loss)
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        
        # 10. Average Days in Trade
        if len(entry_dates) > 0 and len(exit_dates) > 0:
            trade_durations = []
            for i in range(min(len(entry_dates), len(exit_dates))):
                entry_date = pd.to_datetime(entry_dates[i])
                exit_date = pd.to_datetime(exit_dates[i])
                duration = (exit_date - entry_date).days
                trade_durations.append(duration)
            avg_days_in_trade = np.mean(trade_durations) if trade_durations else 0
        else:
            avg_days_in_trade = 0
    else:
        calmar_ratio = None
        sortino_ratio = None
        profit_factor = None
        avg_days_in_trade = 0
    
    strategy_name = 'ML Model'
    stats = f"{strategy_name} Return: {round(calcvalue,2)}%\n" \
            f"Buy & Hold: {round(returnrate,2)}%\n" \
            f"Win Rate: {round(win_rate*100,2):.2f}%\n" \
            f"Total Trades: {len(returns)}\n" \
            f"Avg Return/Trade: {round(avg_return*100,2):.2f}%\n" \
            f"Sharpe Ratio: {round(sharpe,2) if sharpe is not None and not np.isnan(sharpe) else 'N/A'}\n" \
            f"Calmar Ratio: {round(calmar_ratio,2) if calmar_ratio is not None and not np.isnan(calmar_ratio) else 'N/A'}\n" \
            f"Sortino Ratio: {round(sortino_ratio,2) if sortino_ratio is not None and not np.isnan(sortino_ratio) else 'N/A'}\n" \
            f"Profit Factor: {round(profit_factor,2) if profit_factor is not None and not np.isnan(profit_factor) else 'N/A'}\n" \
            f"Avg Days in Trade: {round(avg_days_in_trade,1) if avg_days_in_trade > 0 else 'N/A'}"
    
    # Build performance time series for cumulative returns over the test period only
    performance = [100]
    performance_dates = [str(df_test.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    hold = -1
    
    # Only loop through the test period data
    for i in range(1, len(df_test)):
        # Get predictions for the current test period index
        if i-1 < len(probs_all):
            probs_row = probs_all[i-1]
            predicted_class = np.argmax(probs_row)
            confidence = np.max(probs_row)
        else:
            predicted_class = 1  # Default to hold
            confidence = 0
        
        price_today = df_test['close_series'].iloc[i-1]
        tomorrow_price = df_test['close_series'].iloc[i]
        
        # Entry logic
        if not in_position and predicted_class == 2 and confidence >= buy_confidence:
            in_position = True
            entry_price = tomorrow_price
            entry_portfolio_value = portfolio_value
            hold = -1
        # Exit logic
        elif in_position and hold >= 10 and ((predicted_class == 0 and confidence >= sell_confidence) or price_today <= entry_price - 1.5*df_test['ATR'].iloc[i-1]):
            in_position = False
            exit_price = tomorrow_price
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df_test['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
            hold += 1
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df_test.index[i].date()))
    
    # If still in position at the end, close at last price
    if in_position:
        final_price = df_test['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    # Calculate totalReturn as compounded return
    totalReturn = round((portfolio_value / 100 - 1) * 100, 2)
    
    result = {
        'totalReturn': totalReturn,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if sharpe is not None and not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(calmar_ratio, 2) if calmar_ratio is not None and not np.isnan(calmar_ratio) else None,
        'sortinoRatio': round(sortino_ratio, 2) if sortino_ratio is not None and not np.isnan(sortino_ratio) else None,
        'profitFactor': round(profit_factor, 2) if profit_factor is not None and not np.isnan(profit_factor) else None,
        'avgDaysInTrade': round(avg_days_in_trade, 1) if avg_days_in_trade > 0 else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': ""
    }
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df_test_true
    # Ensure plot_df has the close_series column for plotting
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'ML Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', stats, price_col='close_series')
    
    result['plot_image'] = plot_image
    return result

if __name__ == "__main__":
    run_ml_backtest()



