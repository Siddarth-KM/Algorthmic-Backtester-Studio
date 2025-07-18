import matplotlib
matplotlib.use('Agg')
import pandas as pd
import yfinance as yf
import datetime as dt
from matplotlib import pyplot as plt
import ta
import numpy as np
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD as MACDIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
import runpy  # Add this import to run ML_Project.py as a script
# --- ML-style plotting utility ---
import base64
import io
from typing import Optional

def calculate_advanced_stats(returns, entry_dates, exit_dates, df, final_value=100):
    """Calculate advanced statistics for strategy performance"""
    if not returns or len(returns) < 2:
        return {
            'calmar_ratio': None,
            'sortino_ratio': None,
            'profit_factor': None,
            'avg_days_in_trade': 0
        }
    
    # Calmar Ratio (Annual Return / Max Drawdown)
    cumulative_returns = np.cumprod([1 + r for r in returns])
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
    
    # Annualize the return (assuming daily data, multiply by ~252 trading days)
    days_in_test = (df.index[-1] - df.index[0]).days
    annual_return = (final_value / 100 - 1) * (365 / days_in_test) if days_in_test > 0 else 0
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else None
    
    # Sortino Ratio (Return / Downside Deviation) - need at least 2 negative returns
    negative_returns = [r for r in returns if r < 0]
    if len(negative_returns) >= 2:
        downside_deviation = np.std(negative_returns)
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else None
    else:
        sortino_ratio = None
    
    # Profit Factor (Gross Profit / Gross Loss) - need at least one loss
    gross_profit = sum([r for r in returns if r > 0])
    gross_loss = abs(sum([r for r in returns if r < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    
    # Average Days in Trade
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
    
    return {
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'profit_factor': profit_factor,
        'avg_days_in_trade': avg_days_in_trade
    }

def plot_trades_ml(df, entry_dates, exit_dates, returns, title, stats, price_col='Close'):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # Use dark theme for RWB and ML strategies
    if 'RWB Strategy' in title or 'ML Strategy' in title:
        plt.style.use('dark_background')
        ax.set_facecolor('black')
        plt.gcf().patch.set_facecolor('black')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    # Price line
    plt.plot(df.index, df[price_col], color='tab:gray', label='Stock Price', linewidth=2, zorder=1)
    
    # Buy & hold line
    plt.plot([df.index[0], df.index[-1]],
             [df[price_col].iloc[0], df[price_col].iloc[-1]],
             linestyle='--', color='orange', linewidth=2, label='Buy & Hold Return', zorder=0)
    
    # Blue in-position segments (only if there are trades)
    if entry_dates and exit_dates:
        for entry, exit_, is_win in zip(entry_dates, exit_dates, np.array(returns) > 0):
            if entry in df.index and exit_ in df.index:
                idx_entry = df.index.get_loc(entry)
                idx_exit = df.index.get_loc(exit_)
                segment = df.iloc[idx_entry:idx_exit+1]
                plt.plot(segment.index, segment[price_col], color='royalblue', linewidth=3, alpha=0.7, zorder=2)
    
    # Entry/exit markers (only if there are trades)
    if entry_dates and exit_dates and returns:
        win_mask = np.array(returns) > 0
        loss_mask = ~win_mask
        entry_dates_pd = pd.to_datetime(entry_dates)
        exit_dates_pd = pd.to_datetime(exit_dates)
        entry_prices = df.loc[entry_dates_pd.intersection(df.index), price_col].values
        exit_prices = df.loc[exit_dates_pd.intersection(df.index), price_col].values
        
        # Only plot if we have valid data and masks
        if len(entry_prices) > 0 and len(win_mask) > 0:
            # Plot winning trades
            if np.any(win_mask) and len(entry_prices) >= len(win_mask):
                win_indices = np.where(win_mask)[0]
                if len(win_indices) <= len(entry_prices):
                    plt.scatter(entry_dates_pd[win_indices], entry_prices[win_indices], color='green', marker='^', label='Entry', zorder=5, s=100)
                    plt.scatter(exit_dates_pd[win_indices], exit_prices[win_indices], color='red', marker='v', label='Exit', zorder=5, s=100)
            
            # Plot losing trades
            if np.any(loss_mask) and len(entry_prices) >= len(loss_mask):
                loss_indices = np.where(loss_mask)[0]
                if len(loss_indices) <= len(entry_prices):
                    plt.scatter(entry_dates_pd[loss_indices], entry_prices[loss_indices], color='green', marker='^', zorder=5, s=100, alpha=0.5)
                    plt.scatter(exit_dates_pd[loss_indices], exit_prices[loss_indices], color='red', marker='v', zorder=5, s=100, alpha=0.5)
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend(loc='lower left', fontsize=9, framealpha=0.8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save plot to base64 string for web display - REDUCED DPI for faster processing
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # Close the figure to free memory
    
    return img_base64

def ATR_Stop(df, startyear, stock, df_filtered=None):
    # Flatten columns if MultiIndex (e.g., from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    df['close_series'] = df['Close']
    df['high_series'] = df['High']
    df['low_series'] = df['Low']
    df['ATR'] = AverageTrueRange(high=df['high_series'], low=df['low_series'], close=df['close_series'], window=14).average_true_range()
    for i in range(1, len(df)):
        # Skip if any required value is NaN
        if any(pd.isna([
            df['close_series'].iloc[i-1], df['high_series'].rolling(20).max().shift(1).iloc[i-1],
            df['close_series'].iloc[i], df['high_series'].rolling(20).max().shift(1).iloc[i],
            df['ATR'].iloc[i]
        ])):
            continue
        # Entry: price crosses above 20-day high
        if df['close_series'].iloc[i-1] <= df['high_series'].rolling(20).max().shift(1).iloc[i-1] and df['close_series'].iloc[i] > df['high_series'].rolling(20).max().shift(1).iloc[i] and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            stop = bp - 1.5 * df['ATR'].iloc[i]
            entry_dates.append(df.index[i])
        # Exit: price crosses below trailing stop
        elif pos == 1 and df['close_series'].iloc[i] < stop:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
        # Update stop if in position
        elif pos == 1 and df['close_series'].iloc[i] > df['close_series'].iloc[i-1]:
            stop = df['close_series'].iloc[i] - 1.5 * df['ATR'].iloc[i]
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        calcvalue *= (1 + (i))
    calcvalue-=100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'ATR Stop Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        if df['close_series'].iloc[i-1] <= df['high_series'].rolling(20).max().shift(1).iloc[i-1] and df['close_series'].iloc[i] > df['high_series'].rolling(20).max().shift(1).iloc[i-1] and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and df['close_series'].iloc[i-1] >= df['low_series'].rolling(20).min().shift(1).iloc[i-1] and df['close_series'].iloc[i] < df['low_series'].rolling(20).min().shift(1).iloc[i-1]:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    
    result = {
        'totalReturn': calcvalue,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 2) if advanced_stats['avg_days_in_trade'] is not None and not np.isnan(advanced_stats['avg_days_in_trade']) else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def Bollinger_Band(df, startyear, df_filtered=None):
    # Flatten columns if MultiIndex (e.g., from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    # Create consistent column names
    df['close_series'] = df['Close']
    # Manual calculation of Bollinger Bands
    df['BB_MAVG'] = df['close_series'].rolling(window=20).mean()
    df['BB_STD'] = df['close_series'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MAVG'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MAVG'] - 2 * df['BB_STD']
    for i in range(1, len(df)):
        # Skip if any required value is NaN
        if any(pd.isna([
            df['close_series'].iloc[i-1], df['BB_LOWER'].iloc[i-1],
            df['close_series'].iloc[i], df['BB_LOWER'].iloc[i],
            df['BB_MAVG'].iloc[i-1], df['BB_MAVG'].iloc[i]
        ])):
            continue
        # Buy: price crosses above lower band
        if df['close_series'].iloc[i-1] < df['BB_LOWER'].iloc[i-1] and df['close_series'].iloc[i] >= df['BB_LOWER'].iloc[i] and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: price crosses below moving average (mean reversion exit)
        elif pos == 1 and df['close_series'].iloc[i-1] > df['BB_MAVG'].iloc[i-1] and df['close_series'].iloc[i] <= df['BB_MAVG'].iloc[i]:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        calcvalue *= (1 + (i))
    calcvalue -= 100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    strategy_name = 'Bollinger Band'
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    # Ensure plot_df has the close_series column for plotting
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'Bollinger Band Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        if df['close_series'].iloc[i-1] < df['BB_LOWER'].iloc[i-1] and df['close_series'].iloc[i] >= df['BB_LOWER'].iloc[i] and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and df['close_series'].iloc[i-1] > df['BB_MAVG'].iloc[i-1] and df['close_series'].iloc[i] <= df['BB_MAVG'].iloc[i]:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    # Calculate totalReturn as compounded return
    totalReturn = round((portfolio_value / 100 - 1) * 100, 2)
    
    result = {
        'totalReturn': totalReturn,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 1) if advanced_stats['avg_days_in_trade'] > 0 else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def RSI(df, startyear, stock, df_filtered=None):
    df['close_series'] = df['Close']
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    df['RSI'] = RSIIndicator(close=df['close_series'], window=14).rsi()
    for i in range(1, len(df)):
        # Buy: RSI crosses above 30
        if df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] >= 30 and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: RSI crosses below 70
        elif pos == 1 and df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] <= 70:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        calcvalue *= (1 + (i))
    calcvalue-=100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    strategy_name = 'RSI'
    stats = f"{strategy_name} Return: {round(calcvalue,2)}%\n" \
            f"Buy & Hold: {round(returnrate,2)}%\n" \
            f"Win Rate: {round(win_rate*100,2):.2f}%\n" \
            f"Total Trades: {len(returns)}\n" \
            f"Avg Return/Trade: {round(avg_return*100,2):.2f}%\n" \
            f"Sharpe Ratio: {round(sharpe,2) if not np.isnan(sharpe) else 'N/A'}"
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'RSI Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        if df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] >= 30 and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] <= 70:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)

    result = {
        'totalReturn': calcvalue,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 2) if advanced_stats['avg_days_in_trade'] is not None and not np.isnan(advanced_stats['avg_days_in_trade']) else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': entry_dates,
        'exit_dates': exit_dates,
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def MACD(df, startyear, df_filtered=None):
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    df['close_series'] = df['Close']
    macd_indicator = MACDIndicator(close=df['close_series'], window_slow=30, window_fast=12, window_sign=9)
    df['MACD'] = macd_indicator.macd()
    df['Signal'] = macd_indicator.macd_signal()
    for i in range(1, len(df)):
        # Buy: MACD crosses above signal
        if df['MACD'].iloc[i-1] < df['Signal'].iloc[i-1] and df['MACD'].iloc[i] >= df['Signal'].iloc[i] and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: MACD crosses below signal
        elif pos == 1 and df['MACD'].iloc[i-1] > df['Signal'].iloc[i-1] and df['MACD'].iloc[i] <= df['Signal'].iloc[i]:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        i = float(i)
        calcvalue *= (1 + (i))
    calcvalue -= 100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'MACD Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        if df['MACD'].iloc[i-1] < df['Signal'].iloc[i-1] and df['MACD'].iloc[i] >= df['Signal'].iloc[i] and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and df['MACD'].iloc[i-1] > df['Signal'].iloc[i-1] and df['MACD'].iloc[i] <= df['Signal'].iloc[i]:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    
    result = {
        'totalReturn': calcvalue,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 2) if advanced_stats['avg_days_in_trade'] is not None and not np.isnan(advanced_stats['avg_days_in_trade']) else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': entry_dates,
        'exit_dates': exit_dates,
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def donchian_channel(df, startyear, df_filtered=None):
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    df['close_series'] = df['Close']
    max_val = df['close_series'].rolling(window=20).max().shift(1)
    min_val = df['close_series'].rolling(window=20).min().shift(1)
    for i in range(1, len(df)):
        # Buy: price crosses above 20-day high
        if df['close_series'].iloc[i-1] <= max_val.iloc[i-1] and df['close_series'].iloc[i] > max_val.iloc[i] and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: price crosses below 20-day low
        elif pos == 1 and df['close_series'].iloc[i-1] >= min_val.iloc[i-1] and df['close_series'].iloc[i] < min_val.iloc[i]:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        calcvalue *= (1 + (i))
    calcvalue -= 100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'Donchian Channel Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        if df['close_series'].iloc[i-1] <= max_val.iloc[i-1] and df['close_series'].iloc[i] > max_val.iloc[i] and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and df['close_series'].iloc[i-1] >= min_val.iloc[i-1] and df['close_series'].iloc[i] < min_val.iloc[i]:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    
    result = {
        'totalReturn': calcvalue,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 2) if advanced_stats['avg_days_in_trade'] is not None and not np.isnan(advanced_stats['avg_days_in_trade']) else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def range_breakout(df, startyear, longMA, shortMA, df_filtered=None):
    # Ensure moving average windows are integers
    longMA = int(longMA)
    shortMA = int(shortMA)
    # Create consistent column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    pos = 0
    df['close_series'] = df['Close']
    df['ma1'] = df['close_series'].rolling(window=longMA).mean().shift(1)
    df['ma2'] = df['close_series'].rolling(window=shortMA).mean().shift(1)
    for i in range(1, len(df)):
        ma1_prev = df['ma1'].iloc[i-1]
        ma2_prev = df['ma2'].iloc[i-1]
        ma1 = df['ma1'].iloc[i]
        ma2 = df['ma2'].iloc[i]
        # Buy: short MA crosses above long MA
        if ma2_prev <= ma1_prev and ma2 > ma1 and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: short MA crosses below long MA
        elif pos == 1 and ma2_prev >= ma1_prev and ma2 < ma1:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float((sp/bp-1))
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos == 1:
        sp = df['close_series'].iloc[-1]
        pc = float((sp/bp-1))
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        calcvalue *= (1 + (i))
    calcvalue -= 100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'Range Breakout Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        ma1_prev = df['ma1'].iloc[i-1]
        ma2_prev = df['ma2'].iloc[i-1]
        ma1 = df['ma1'].iloc[i]
        ma2 = df['ma2'].iloc[i]
        # Entry logic
        if ma2_prev <= ma1_prev and ma2 > ma1 and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        # Exit logic
        elif in_position and ma2_prev >= ma1_prev and ma2 < ma1:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    
    result = {
        'totalReturn': calcvalue,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 2) if advanced_stats['avg_days_in_trade'] is not None and not np.isnan(advanced_stats['avg_days_in_trade']) else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def RWB_strategy(df, startyear, df_filtered=None):
    # Create consistent column names
    df['close_series'] = df['Close']
    emasUsed = [3,5,8,10,12,15,30,35,40,45,50,60]
    for x in emasUsed:
        ema = x
        df["Ema_" + str(ema)] = round(df['close_series'].ewm(span=ema, adjust=False).mean(),2)
    pos = 0
    percentchange = []
    entry_dates = []
    exit_dates = []
    returns = []
    for i in range(1, len(df)):
        cmin_prev = min(df["Ema_3"].iloc[i-1],df["Ema_5"].iloc[i-1],df["Ema_8"].iloc[i-1],df["Ema_10"].iloc[i-1],df["Ema_12"].iloc[i-1],df["Ema_15"].iloc[i-1])
        cmax_prev = max(df["Ema_30"].iloc[i-1],df["Ema_35"].iloc[i-1],df["Ema_40"].iloc[i-1],df["Ema_45"].iloc[i-1],df["Ema_50"].iloc[i-1],df["Ema_60"].iloc[i-1])
        cmin = min(df["Ema_3"].iloc[i],df["Ema_5"].iloc[i],df["Ema_8"].iloc[i],df["Ema_10"].iloc[i],df["Ema_12"].iloc[i],df["Ema_15"].iloc[i])
        cmax = max(df["Ema_30"].iloc[i],df["Ema_35"].iloc[i],df["Ema_40"].iloc[i],df["Ema_45"].iloc[i],df["Ema_50"].iloc[i],df["Ema_60"].iloc[i])
        # Buy: short EMAs cross above long EMAs
        if cmin_prev <= cmax_prev and cmin > cmax and pos == 0:
            pos = 1
            bp = df['close_series'].iloc[i]
            entry_dates.append(df.index[i])
        # Sell: short EMAs cross below long EMAs
        elif pos == 1 and cmin_prev >= cmax_prev and cmin < cmax:
            pos = 0
            sp = df['close_series'].iloc[i]
            pc = float(sp/bp-1)
            percentchange.append(pc)
            returns.append(pc)
            exit_dates.append(df.index[i])
    if pos==1:
        pos = 0
        sp = df['close_series'].iloc[-1]
        pc = float(sp/bp-1)
        percentchange.append(pc)
        returns.append(pc)
        exit_dates.append(df.index[-1])
    calcvalue = 100
    for i in percentchange:
        i = float(i)
        calcvalue *= (1 + i)
    calcvalue-=100
    firstclose = df['close_series'].iloc[0]
    finalclose = df['close_series'].iloc[-1]
    returnrate = round(((finalclose - firstclose) / firstclose) * 100, 2)
    avg_return = np.mean(returns) if returns else 0
    win_rate = np.mean([1 if r > 0 else 0 for r in returns]) if returns else 0
    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else np.nan
    # Calculate advanced stats
    advanced_stats = calculate_advanced_stats(returns, entry_dates, exit_dates, df, calcvalue + 100)
    
    # Use filtered DataFrame for plotting if provided, otherwise use full DataFrame
    plot_df = df_filtered if df_filtered is not None else df
    if plot_df is not None and 'close_series' not in plot_df.columns:
        plot_df['close_series'] = plot_df['Close']
    plot_image = plot_trades_ml(plot_df, entry_dates, exit_dates, returns, 'RWB Strategy - Stock Price with Trade Entries, Exits & In-Position Highlighted', '', price_col='close_series')
    
    # Build performance time series for cumulative returns over the entire period
    performance = [100]
    performance_dates = [str(df.index[0].date())]
    portfolio_value = 100
    in_position = False
    entry_price = 0
    entry_portfolio_value = 100
    for i in range(1, len(df)):
        # Entry logic
        cmin_prev = min(df["Ema_3"].iloc[i-1],df["Ema_5"].iloc[i-1],df["Ema_8"].iloc[i-1],df["Ema_10"].iloc[i-1],df["Ema_12"].iloc[i-1],df["Ema_15"].iloc[i-1])
        cmax_prev = max(df["Ema_30"].iloc[i-1],df["Ema_35"].iloc[i-1],df["Ema_40"].iloc[i-1],df["Ema_45"].iloc[i-1],df["Ema_50"].iloc[i-1],df["Ema_60"].iloc[i-1])
        cmin = min(df["Ema_3"].iloc[i],df["Ema_5"].iloc[i],df["Ema_8"].iloc[i],df["Ema_10"].iloc[i],df["Ema_12"].iloc[i],df["Ema_15"].iloc[i])
        cmax = max(df["Ema_30"].iloc[i],df["Ema_35"].iloc[i],df["Ema_40"].iloc[i],df["Ema_45"].iloc[i],df["Ema_50"].iloc[i],df["Ema_60"].iloc[i])
        
        # Buy signal
        if cmin_prev <= cmax_prev and cmin > cmax and not in_position:
            in_position = True
            entry_price = df['close_series'].iloc[i]
            entry_portfolio_value = portfolio_value
        
        # Sell signal
        elif in_position and cmin_prev >= cmax_prev and cmin < cmax:
            in_position = False
            exit_price = df['close_series'].iloc[i]
            trade_return = (exit_price / entry_price) - 1
            portfolio_value = entry_portfolio_value * (1 + trade_return)
        
        # Mark-to-market logic
        if in_position:
            current_price = df['close_series'].iloc[i]
            unrealized_return = (current_price / entry_price) - 1
            performance.append(entry_portfolio_value * (1 + unrealized_return))
        else:
            performance.append(portfolio_value)
        performance_dates.append(str(df.index[i].date()))
    # If still in position at the end, close at last price
    if in_position:
        final_price = df['close_series'].iloc[-1]
        final_return = (final_price / entry_price) - 1
        portfolio_value = entry_portfolio_value * (1 + final_return)
        performance[-1] = float(portfolio_value)
    # Calculate totalReturn as compounded return
    totalReturn = round((portfolio_value / 100 - 1) * 100, 2)
    
    result = {
        'totalReturn': totalReturn,
        'winRate': round(win_rate*100, 2),
        'trades': len(returns),
        'sharpeRatio': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'buyHoldReturn': round(returnrate, 2),
        'avgReturnPerTrade': round(avg_return*100, 2),
        'calmarRatio': round(advanced_stats['calmar_ratio'], 2) if advanced_stats['calmar_ratio'] is not None and not np.isnan(advanced_stats['calmar_ratio']) else None,
        'sortinoRatio': round(advanced_stats['sortino_ratio'], 2) if advanced_stats['sortino_ratio'] is not None and not np.isnan(advanced_stats['sortino_ratio']) else None,
        'profitFactor': round(advanced_stats['profit_factor'], 2) if advanced_stats['profit_factor'] is not None and not np.isnan(advanced_stats['profit_factor']) else None,
        'avgDaysInTrade': round(advanced_stats['avg_days_in_trade'], 1) if advanced_stats['avg_days_in_trade'] > 0 else None,
        'performance': performance,
        'performance_dates': performance_dates,
        'entry_dates': [str(d) for d in entry_dates],
        'exit_dates': [str(d) for d in exit_dates],
        'returns': returns,
        'plot_image': plot_image
    }
    return result

def get_latest_valid_close(df, close_col):
    """Helper function to find the most recent valid close price"""
    for i in range(-1, -len(df), -1):  # Start from -1 (last row) and go backwards
        try:
            close_val = float(close_col.iloc[i])
            if not pd.isna(close_val) and close_val > 0:
                return close_val, i
        except (ValueError, IndexError):
            continue
    return None, None

def RWB_daily_signal(df):
    if len(df) < 61:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 61 days)"}
    emasUsed = [3,5,8,10,12,15,30,35,40,45,50,60]
    for x in emasUsed:
        df["Ema_" + str(x)] = round(df['Close'].ewm(span=x, adjust=False).mean(),2)
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        cmin_prev = min(float(df["Ema_3"].iloc[close_idx_2]),float(df["Ema_5"].iloc[close_idx_2]),float(df["Ema_8"].iloc[close_idx_2]),float(df["Ema_10"].iloc[close_idx_2]),float(df["Ema_12"].iloc[close_idx_2]),float(df["Ema_15"].iloc[close_idx_2]))
        cmax_prev = max(float(df["Ema_30"].iloc[close_idx_2]),float(df["Ema_35"].iloc[close_idx_2]),float(df["Ema_40"].iloc[close_idx_2]),float(df["Ema_45"].iloc[close_idx_2]),float(df["Ema_50"].iloc[close_idx_2]),float(df["Ema_60"].iloc[close_idx_2]))
        cmin = min(float(df["Ema_3"].iloc[close_idx]),float(df["Ema_5"].iloc[close_idx]),float(df["Ema_8"].iloc[close_idx]),float(df["Ema_10"].iloc[close_idx]),float(df["Ema_12"].iloc[close_idx]),float(df["Ema_15"].iloc[close_idx]))
        cmax = max(float(df["Ema_30"].iloc[close_idx]),float(df["Ema_35"].iloc[close_idx]),float(df["Ema_40"].iloc[close_idx]),float(df["Ema_45"].iloc[close_idx]),float(df["Ema_50"].iloc[close_idx]),float(df["Ema_60"].iloc[close_idx]))
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for RWB calculation"}
    
    if cmin_prev <= cmax_prev and cmin > cmax:
        signal = "BUY"
        reason = "Reason: Short EMAs crossed above long EMAs"
    elif cmin_prev >= cmax_prev and cmin < cmax:
        signal = "SELL"
        reason = "Reason: Short EMAs crossed below long EMAs"
    else:
        signal = "HOLD"
        reason = "Reason: No crossover detected"
    return {
        "signal": signal,
        "close_price": close_prev_2,
        "reason": reason,
        "short_ema_min": round(float(cmin), 2),
        "long_ema_max": round(float(cmax), 2)
    }

def ATR_Stop_daily_signal(df):
    if len(df) < 21:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 21 days)"}
    # Handle MultiIndex columns from yfinance
    high_col = df['High'].iloc[:, 0] if df['High'].ndim > 1 else df['High']
    low_col = df['Low'].iloc[:, 0] if df['Low'].ndim > 1 else df['Low']
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    
    ATR = np.asarray(AverageTrueRange(high=high_col, low=low_col, close=close_col, window=14).average_true_range()).ravel()
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    high_20_prev = np.asarray(high_col.rolling(20).max().shift(1)).ravel()
    try:
        atr_prev = float(ATR[close_idx_2])
        stop = close_prev_2 - 1.5 * atr_prev
        close_val = float(close_col.iloc[close_idx])
        high_20_prev_val = float(high_20_prev[close_idx_2])
        high_20_now_val = float(high_20_prev[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for ATR Stop calculation"}
    if close_prev_2 <= high_20_prev_val and close_val > high_20_now_val:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: Price crossed above 20-day high"}
    elif stop is not None and close_val < stop:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: Price crossed below trailing stop"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover detected"}

def RangeBreakout_daily_signal(df, longMA=50, shortMA=20):
    if len(df) < max(longMA, shortMA) + 1:
        return {"signal": "HOLD", "close_price": None, "reason": f"Reason: Insufficient data (need at least {max(longMA, shortMA)+1} days)"}
    smalString = 'Sma_' + str(int(longMA))
    smasString = 'Sma_' + str(int(shortMA))
    df[smalString] = df['Close'].rolling(window=int(longMA)).mean()
    df[smasString] = df['Close'].rolling(window=int(shortMA)).mean()
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        ma1_prev = float(df[smalString].iloc[close_idx_2])
        ma2_prev = float(df[smasString].iloc[close_idx_2])
        ma1 = float(df[smalString].iloc[close_idx])
        ma2 = float(df[smasString].iloc[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for Range Breakout calculation"}
    if ma2_prev <= ma1_prev and ma2 > ma1:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: Short MA crossed above long MA"}
    elif ma2_prev >= ma1_prev and ma2 < ma1:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: Short MA crossed below long MA"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover"}

def Donchian_daily_signal(df):
    if len(df) < 21:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 21 days)"}
    max_val = np.asarray(df['Close'].rolling(window=20).max().shift(1)).ravel()
    min_val = np.asarray(df['Close'].rolling(window=20).min().shift(1)).ravel()
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        close_val = float(close_col.iloc[close_idx])
        max_val_prev = float(max_val[close_idx_2])
        max_val_now = float(max_val[close_idx])
        min_val_prev = float(min_val[close_idx_2])
        min_val_now = float(min_val[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for Donchian Channel calculation"}
    if close_prev_2 <= max_val_prev and close_val > max_val_now:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: Price crossed above 20-day high"}
    elif close_prev_2 >= min_val_prev and close_val < min_val_now:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: Price crossed below 20-day low"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover detected"}

def MACD_daily_signal(df):
    if len(df) < 36:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 36 days)"}
    # Handle MultiIndex columns from yfinance
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    
    macd_indicator = MACDIndicator(close=close_col, window_slow=30, window_fast=12, window_sign=9)
    MACD = np.asarray(macd_indicator.macd()).ravel()
    Signal = np.asarray(macd_indicator.macd_signal()).ravel()
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        MACD_prev = float(MACD[close_idx_2])
        MACD_now = float(MACD[close_idx])
        Signal_prev = float(Signal[close_idx_2])
        Signal_now = float(Signal[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for MACD calculation"}
    if MACD_prev < Signal_prev and MACD_now >= Signal_now:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: MACD crossed above signal"}
    elif MACD_prev > Signal_prev and MACD_now <= Signal_now:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: MACD crossed below signal"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover detected"}

def RSI_daily_signal(df):
    if len(df) < 16:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 16 days)"}
    # Handle MultiIndex columns from yfinance
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    
    RSI = np.asarray(RSIIndicator(close=close_col, window=14).rsi()).ravel()
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        RSI_prev = float(RSI[close_idx_2])
        RSI_now = float(RSI[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for RSI calculation"}
    if RSI_prev < 30 and RSI_now >= 30:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: RSI crossed above 30"}
    elif RSI_prev > 70 and RSI_now <= 70:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: RSI crossed below 70"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover detected"}

def BollingerBand_daily_signal(df):
    if len(df) < 21:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 21 days)"}
    BB_MAVG = np.asarray(df['Close'].rolling(window=20).mean()).ravel()
    BB_STD = np.asarray(df['Close'].rolling(window=20).std()).ravel()
    BB_UPPER = BB_MAVG + 2 * BB_STD
    BB_LOWER = BB_MAVG - 2 * BB_STD
    if len(df) < 2:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 2 rows)"}
    
    # Get the most recent valid close price
    close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    close_prev, close_idx = get_latest_valid_close(df, close_col)
    if close_prev is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
    
    # Get the second most recent valid close price
    close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
    if close_prev_2 is None:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
    
    try:
        if (
            pd.isna(BB_MAVG[close_idx]) or pd.isna(BB_MAVG[close_idx_2]) or
            pd.isna(BB_STD[close_idx]) or pd.isna(BB_STD[close_idx_2]) or
            pd.isna(BB_UPPER[close_idx]) or pd.isna(BB_UPPER[close_idx_2]) or
            pd.isna(BB_LOWER[close_idx]) or pd.isna(BB_LOWER[close_idx_2])
        ):
            raise ValueError("NaN in rolling window")
        close_val = float(close_col.iloc[close_idx])
        BB_LOWER_prev = float(BB_LOWER[close_idx_2])
        BB_LOWER_now = float(BB_LOWER[close_idx])
        BB_MAVG_prev = float(BB_MAVG[close_idx_2])
        BB_MAVG_now = float(BB_MAVG[close_idx])
    except Exception:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient or NaN data for Bollinger Bands calculation"}
    if close_prev_2 < BB_LOWER_prev and close_val >= BB_LOWER_now:
        return {"signal": "BUY", "close_price": close_prev_2, "reason": "Reason: Price crossed above lower band"}
    elif close_prev_2 > BB_MAVG_prev and close_val <= BB_MAVG_now:
        return {"signal": "SELL", "close_price": close_prev_2, "reason": "Reason: Price crossed below moving average"}
    else:
        return {"signal": "HOLD", "close_price": close_prev_2, "reason": "Reason: No crossover detected"}

def ML_daily_signal(df, buy_confidence=0.5, sell_confidence=0.9):
    if len(df) < 61:
        return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data (need at least 61 days)"}
    
    try:
        # Handle MultiIndex columns from yfinance
        close_col = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
        high_col = df['High'].iloc[:, 0] if df['High'].ndim > 1 else df['High']
        low_col = df['Low'].iloc[:, 0] if df['Low'].ndim > 1 else df['Low']
        volume_col = df['Volume'].iloc[:, 0] if df['Volume'].ndim > 1 else df['Volume']
        
        # Create a copy for calculations
        df_calc = df.copy()
        df_calc['close_series'] = close_col
        df_calc['high_series'] = high_col
        df_calc['low_series'] = low_col
        df_calc['volume_series'] = volume_col
        
        # Calculate all indicators (same as in ML_Project.py)
        from scipy.stats import linregress
        
        def calc_slope(series):
            if series.isna().any():
                return np.nan
            x = np.arange(len(series))
            slope, _, _, _, _ = linregress(x, series)
            return float(slope)
        
        df_calc['slope_price_10d'] = df_calc['close_series'].rolling(window=10).apply(calc_slope, raw=False)
        df_calc['ATR'] = AverageTrueRange(high=df_calc['high_series'], low=df_calc['low_series'], close=df_calc['close_series']).average_true_range()
        df_calc['SMA_20'] = df_calc['close_series'].rolling(window=20).mean().shift(1)
        df_calc['SD'] = df_calc['close_series'].rolling(window=20).std().shift(1)
        df_calc['Upper'] = df_calc['SMA_20'] + 2*df_calc['SD']
        df_calc['Lower'] = df_calc['SMA_20'] - 2*df_calc['SD']
        df_calc['width'] = (df_calc['Upper'] - df_calc['Lower']) / df_calc['SMA_20']
        df_calc['percent_b'] = (df_calc['close_series'] - df_calc['Lower']) / (df_calc['Upper'] - df_calc['Lower'])
        rolling_mean = df_calc['close_series'].rolling(window=10).mean()
        df_calc['rolling_std_10'] = df_calc['close_series'].rolling(window=10).std()
        df_calc['z_score_close'] = (df_calc['close_series'] - rolling_mean) / df_calc['rolling_std_10']
        df_calc["Ema_10"] = df_calc['Close'].ewm(span=10, adjust=False).mean()
        df_calc['volume_pct_change'] = df_calc['Volume'].pct_change()
        df_calc['RSI'] = RSIIndicator(close=df_calc['close_series']).rsi()
        df_calc['roc_10'] = ROCIndicator(close=df_calc['close_series']).roc()
        df_calc['momentum'] = df_calc['close_series'] - df_calc['close_series'].shift(10)
        df_calc['ema_diff'] = df_calc['Ema_10'] - df_calc['SMA_20']
        df_calc['price_position'] = (df_calc['close_series'] - df_calc['Lower']) / (df_calc['Upper'] - df_calc['Lower'])
        df_calc['past_10d_return'] = df_calc['close_series'] / df_calc['close_series'].shift(10) - 1
        df_calc['rsi_14_diff'] = df_calc['RSI'] - df_calc['RSI'].shift(5)
        df_calc['price_vs_ema'] = df_calc['close_series'] - df_calc['Ema_10']
        df_calc['adx_14'] = ADXIndicator(high=df_calc['high_series'], low=df_calc['low_series'], close=df_calc['close_series']).adx()
        df_calc['OBV'] = OnBalanceVolumeIndicator(close=df_calc['close_series'], volume=df_calc['volume_series']).on_balance_volume()
        df_calc['log_return'] = np.log(df_calc['close_series'] / df_calc['close_series'].shift(1))
        df_calc['volatility'] = df_calc['log_return'].rolling(10).std()
        df_calc['ema_ratio'] = df_calc['Ema_10'] / df_calc['SMA_20']
        
        for lag in range(1, 6):
            df_calc[f'close_lag_{lag}'] = df_calc['Close'].shift(lag)
        
        df_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_calc.dropna(inplace=True)
        
        if len(df_calc) < 20:
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient data after indicator calculation"}
        
        # Define indicators list
        indicators = ['width', 'percent_b', 'z_score_close', 'rolling_std_10', 'slope_price_10d','log_return', 'ema_ratio', 'RSI' ,'SMA_20', 'Ema_10', 'roc_10','Upper', 'Lower', 'ATR', 'volume_pct_change', 'ema_diff' , 'price_position', 'past_10d_return','rsi_14_diff','momentum','price_vs_ema', 'volatility','adx_14', 'OBV', 'close_lag_1'  ,'close_lag_2' ,'close_lag_3' ,'close_lag_4' ,'close_lag_5']
        
        # Use last 60 days for training, last day for prediction
        train_data = df_calc.iloc[:-1]  # All but last day
        predict_data = df_calc.iloc[-1:]  # Last day only
        
        if len(train_data) < 30:
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient training data"}
        
        # Prepare training data
        train_data['future_return'] = train_data['close_series'].shift(-10) / train_data['close_series'] - 1
        train_data['Label'] = 1
        train_data.loc[train_data['future_return'] > 0.015, 'Label'] = 2
        train_data.loc[train_data['future_return'] < -0.025, 'Label'] = 0
        
        # Balance classes
        from sklearn.utils import class_weight, resample
        df_train_ml = train_data[indicators + ['Label']].dropna()
        df_0 = df_train_ml[df_train_ml['Label'] == 0]
        df_1 = df_train_ml[df_train_ml['Label'] == 1]
        df_2 = df_train_ml[df_train_ml['Label'] == 2]
        
        if len(df_0) == 0 or len(df_1) == 0 or len(df_2) == 0:
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: Insufficient class diversity in training data"}
        
        minority_size = min(len(df_0), len(df_1), len(df_2))
        df_0 = resample(df_0, replace=False, n_samples=minority_size, random_state=42)
        df_1 = resample(df_1, replace=False, n_samples=minority_size, random_state=42)
        df_2 = resample(df_2, replace=False, n_samples=minority_size, random_state=42)
        df_balanced = pd.concat([df_0, df_1, df_2]).sample(frac=1, random_state=42)
        
        x_train = df_balanced[indicators]
        y_train = df_balanced['Label']
        
        # Train model
        from xgboost import XGBClassifier
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
        
        # Predict on last day
        x_predict = predict_data[indicators]
        if x_predict.isna().any().any():
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: Missing indicator values for prediction"}
        
        probs = model.predict_proba(x_predict)
        predicted_class = np.argmax(probs[0])
        confidence = np.max(probs[0])
        
        # Get the most recent valid close price
        close_prev, close_idx = get_latest_valid_close(df, close_col)
        if close_prev is None:
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: No valid close price found"}
        
        # Get the second most recent valid close price for the signal
        close_prev_2, close_idx_2 = get_latest_valid_close(df.iloc[:close_idx], close_col.iloc[:close_idx])
        if close_prev_2 is None:
            return {"signal": "HOLD", "close_price": None, "reason": "Reason: No second valid close price found"}
        
        # Get all 3 probabilities
        class_names = ["SELL", "HOLD", "BUY"]
        probabilities = probs[0]
        class_prob_pairs = list(zip(range(len(class_names)), probabilities))
        class_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        all_probabilities = []
        for class_idx, prob in class_prob_pairs:
            all_probabilities.append({
                "class": class_names[class_idx],
                "probability": round(float(prob), 3)
            })
        
        # Determine signal based on prediction and confidence
        if predicted_class == 2 and confidence >= buy_confidence:
            signal = "BUY"
        elif predicted_class == 0 and confidence >= sell_confidence:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return {
            "signal": signal,
            "close_price": close_prev_2,
            "confidence": round(float(confidence), 3),
            "predicted_class": int(predicted_class),
            "all_probabilities": all_probabilities
        }
        
    except Exception as e:
        return {"signal": "HOLD", "close_price": None, "reason": f"Reason: ML calculation error: {str(e)}"}

# --- CLI and script-running code ---
if __name__ == "__main__":
    # Display a clean, numbered list of strategies
    strategies = [
        'ATR_Stop',
        'RWB',
        'Range Breakout',
        'Donchian Channel',
        'MACD',
        'RSI',
        'Bollinger Band Breakout',
        'ML Strategy (XGBoost)'
    ]
    print("\nSelect an investing strategy to test:")
    for i, strat_name in enumerate(strategies, 1):
        if strat_name == '---':
            continue
        print(f"  {i}. {strat_name}")
    while True:
        try:
            strat_num = int(input("Enter the number of the strategy you want to test (1-8): "))
            if strat_num in list(range(1,9)):
                break
            else:
                print("Please enter a number from 1 to 8.")
        except ValueError:
            print("Invalid input. Please enter a number from 1 to 8.")
    strat = str(strat_num)

    if strat == '8':
        # Import and call the ML strategy as a function, preserving CLI in ML_Project.py
        from ML_Project import run_ml_backtest
        run_ml_backtest()
        exit()

    stock = input('Enter stock symbol: ')
    stock = stock.upper()
    print("Select a start date:")
    print("1. 1 month")
    print("2. 2 months")
    print("3. 3 months")
    print("4. 4 months")
    print("5. 5 months")
    print("6. 6 months")
    print("7. Custom (enter specific date)")
    while True:
        start_choice = input("Enter choice (1-7): ")
        if start_choice in [str(i) for i in range(1,8)]:
            break
        else:
            print("Please enter a number from 1 to 7.")
    end = (dt.datetime.now() - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    if start_choice == '1':
        start = end - pd.DateOffset(months=1)
    elif start_choice == '2':
        start = end - pd.DateOffset(months=2)
    elif start_choice == '3':
        start = end - pd.DateOffset(months=3)
    elif start_choice == '4':
        start = end - pd.DateOffset(months=4)
    elif start_choice == '5':
        start = end - pd.DateOffset(months=5)
    elif start_choice == '6':
        start = end - pd.DateOffset(months=6)
    elif start_choice == '7':
        while True:
            custom_date = input("Enter the start date (YYYY-MM-DD): ")
            try:
                start = pd.to_datetime(custom_date)
                break
            except Exception:
                print("Invalid date format. Please enter as YYYY-MM-DD.")

    df = yf.download(stock, start=start, end=end) # Pulls the data needed from yahoo finance
    firstopen = df.iloc[0,3] # Sets the original buy price for no stargey
    finalclose = df.iloc[-1,0] # Sets the final sell price for no strategy
    returnrate = round(((finalclose - firstopen)/firstopen)*100,2) # Calculates the retrun rate if the stock was just bought and kept wuth no strategy
    if strat == '2':
        calcvalue = RWB_strategy(df, None)
    elif strat == '1':
        calcvalue = ATR_Stop(df, None, stock)
    elif strat == '3':
        calcvalue = range_breakout(df, None, None, None)
    elif strat == '4':
        calcvalue = donchian_channel(df, None)
    elif strat=='5':
        calcvalue = MACD(df, None)
    elif strat =='6':
        calcvalue = RSI(df, None, stock)
    elif strat == '7':
        calcvalue = Bollinger_Band(df, None)
    else:
        calcvalue = None
        print('Other options under development')

    # Only process output if calcvalue is defined
    if calcvalue is not None:
        if isinstance(calcvalue, dict):
            total_return = calcvalue['totalReturn']
            buy_hold_return = calcvalue['buyHoldReturn'] if 'buyHoldReturn' in calcvalue else returnrate
            stats = calcvalue['stats'] if 'stats' in calcvalue else ''
        else:
            total_return = calcvalue
            buy_hold_return = returnrate
            stats = ''

        print(f"If you had just invested the money and left it alone, you would have made a {buy_hold_return} percent return")
        # Use a mapping for clean strategy names
        strategy_names = {
            '1': 'ATR_Stop',
            '2': 'RWB',
            '3': 'Range Breakout',
            '4': 'Donchian Channel',
            '5': 'MACD',
            '6': 'RSI',
            '7': 'Bollinger Band Breakout'
        }
        strat_name = strategy_names.get(strat, 'Unknown Strategy')

        if float(total_return) > float(buy_hold_return):
            print(f"{strat_name} could be a smart idea!!!")
        else:
            print(f"{strat_name} wouldn't be smart right now, you'd probably make more money just investing the money and leaving it.")

        # Optionally print the stats box for full consistency with the graph
        print(stats)

# --- API-friendly function for backend ---
def run_standard_backtest(strategy, ticker, testPeriod, customStart, longMA=None, shortMA=None):
    strategy_map = {
        '1': ATR_Stop,
        '2': RWB_strategy,
        '3': range_breakout,
        '4': donchian_channel,
        '5': MACD,
        '6': RSI,
        '7': Bollinger_Band
    }
    end = (dt.datetime.now() - pd.Timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=0)
    # Determine user-requested start date
    if testPeriod == '1':  # 4 days
        user_start = end - pd.DateOffset(days=4)
    elif testPeriod == '2':  # 1 week
        user_start = end - pd.DateOffset(days=7)
    elif testPeriod == '3':  # 2 weeks (14 calendar days)
        user_start = end - pd.DateOffset(days=14)
    elif testPeriod == '4':  # 1 month
        user_start = end - pd.DateOffset(months=1)
    elif testPeriod == '5':  # 3 months
        user_start = end - pd.DateOffset(months=3)
    elif testPeriod == '6':  # 6 months
        user_start = end - pd.DateOffset(months=6)
    elif testPeriod == '7' and customStart:
        try:
            user_start = pd.to_datetime(customStart)
            if not isinstance(user_start, pd.Timestamp):
                user_start = pd.Timestamp(user_start)
        except Exception:
            user_start = end - pd.DateOffset(months=1)
    else:
        user_start = end - pd.DateOffset(months=1)
    
    # Calculate how much extra data we need for indicators
    # Most strategies use 20-day windows, some use 14-day, so we need at least 30 days extra
    extra_days_needed = 30
    
    # Fetch data starting from user_start minus extra days needed
    fetch_start = user_start - pd.DateOffset(days=extra_days_needed)
    
    # Download data with error handling (same as ML strategy)
    df: Optional[pd.DataFrame] = yf.download(ticker, start=fetch_start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"Failed to download data for {ticker}. Please check the ticker symbol and try again.")
        return {"error": f"Failed to download data for {ticker}"}
    
    # Ensure df is a proper DataFrame with DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"Warning: DataFrame index is not DatetimeIndex, converting...")
        df.index = pd.to_datetime(df.index)
    
    # Ensure user_start is a pd.Timestamp for comparison
    if not isinstance(user_start, pd.Timestamp):
        user_start = pd.Timestamp(user_start)
    df_filtered = df[df.index >= user_start].copy()
    
    # Run the strategy on the full dataset (including the extra days for indicators)
    func = strategy_map.get(str(strategy))
    if not func:
        return {"error": "Invalid or unsupported strategy"}
    
    if func == ATR_Stop or func == RSI:
        result = func(df, None, ticker, df_filtered)
    elif func == range_breakout:
        if longMA is None or shortMA is None:
            return {"error": "Missing longMA or shortMA for Range Breakout"}
        result = func(df, None, longMA, shortMA, df_filtered)
    else:
        result = func(df, None, df_filtered)
    
    # Filter the result to only include data from the user's requested period
    if isinstance(result, dict) and 'performance' in result and 'performance_dates' in result:
        # Find the index where the user-requested period starts
        user_start_str = user_start.strftime('%Y-%m-%d')
        start_idx = 0
        for i, date_str in enumerate(result['performance_dates']):
            if date_str >= user_start_str:
                start_idx = i
                break
        
        # Filter performance data to only include user-requested period
        result['performance'] = result['performance'][start_idx:]
        result['performance_dates'] = result['performance_dates'][start_idx:]
        
        # Also filter entry/exit dates to only include those within the user period
        if 'entry_dates' in result and 'exit_dates' in result:
            filtered_entry_dates = []
            filtered_exit_dates = []
            filtered_returns = []
            
            for i, entry_date in enumerate(result['entry_dates']):
                if entry_date >= user_start_str:
                    filtered_entry_dates.append(entry_date)
                    filtered_exit_dates.append(result['exit_dates'][i])
                    filtered_returns.append(result['returns'][i])
            
            result['entry_dates'] = filtered_entry_dates
            result['exit_dates'] = filtered_exit_dates
            result['returns'] = filtered_returns
            result['trades'] = len(filtered_returns)
            
            # Recalculate statistics based on filtered trades
            if filtered_returns:
                result['totalReturn'] = round((np.prod([1 + r for r in filtered_returns]) - 1) * 100, 2)
                result['winRate'] = round(np.mean([1 if r > 0 else 0 for r in filtered_returns]) * 100, 2)
                result['avgReturnPerTrade'] = round(np.mean(filtered_returns) * 100, 2)
                if len(filtered_returns) > 1:
                    result['sharpeRatio'] = round(np.mean(filtered_returns) / np.std(filtered_returns), 2)
    
    return result
