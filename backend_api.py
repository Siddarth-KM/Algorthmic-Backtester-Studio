from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime as dt
import pandas as pd
import yfinance as yf
from backtester_main import run_standard_backtest
from ML_Project import run_ml_backtest
import logging

logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')

app = Flask(__name__)
CORS(app)

def run_standard_strategy(strategy, ticker, testPeriod, customStart, longMA=None, shortMA=None):
    # Call the real backtesting logic
    try:
        if strategy == '3':
            result = run_standard_backtest(strategy, ticker, testPeriod, customStart, longMA, shortMA)
        else:
            result = run_standard_backtest(strategy, ticker, testPeriod, customStart)
        return result
    except Exception as e:
        return {"error": str(e)}

def run_ml_strategy(ticker, download_start_date, test_period, buy_confidence=0.5, sell_confidence=0.9):
    try:
        end = dt.datetime.now()
        # Use dictionary for test_period mapping, but keep logic identical
        def get_user_start(period):
            if period == '1':
                return end - pd.DateOffset(months=3)
            elif period == '2':
                return end - pd.DateOffset(months=6)
            elif period == '3':
                return pd.Timestamp(end.year, 1, 1)
            elif period == '4':
                return end - pd.DateOffset(years=1)
            elif period == '5':
                return end - pd.DateOffset(years=2)
            else:
                return end - pd.DateOffset(years=1)
        user_start = get_user_start(test_period)
        df_filtered = yf.download(ticker, start=user_start, end=end)
        if df_filtered is None or df_filtered.empty:
            return {"error": f"No data found for {ticker}"}
        result = run_ml_backtest(
            ticker=ticker,
            download_start_date=download_start_date,
            test_period=test_period,
            buy_confidence=buy_confidence,
            sell_confidence=sell_confidence,
            df_filtered=df_filtered
        )
        return result
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"})
        
        strategy = data.get('strategy')
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({"error": "Ticker is required"})
        if strategy == '9':
            download_start_date = data.get('mlDownloadStart')
            test_period = data.get('mlTestPeriod')
            buy_confidence = float(data.get('buyConfidence', 0.5))
            sell_confidence = float(data.get('sellConfidence', 0.9))
            result = run_ml_strategy(ticker, download_start_date, test_period, buy_confidence, sell_confidence)
        else:
            testPeriod = data.get('testPeriod')
            customStart = data.get('customStart')
            longMA = data.get('longMA')
            shortMA = data.get('shortMA')
            result = run_standard_strategy(strategy, ticker, testPeriod, customStart, longMA, shortMA)
        
        return jsonify(result)
    except Exception as e:
        logging.error('Error in /api/backtest', exc_info=True)
        return jsonify({"error": f"Backend error: {str(e)}"}), 500

@app.route('/api/daily-signal/<strategy>', methods=['GET', 'POST'])
def api_daily_signal(strategy):
    if request.method == 'POST':
        data = request.get_json()
        ticker = data.get('ticker')
        longMA = data.get('longMA')
        shortMA = data.get('shortMA')
    else:
        ticker = request.args.get('ticker')
        longMA = request.args.get('longMA')
        shortMA = request.args.get('shortMA')
    if not ticker:
        return jsonify({"signal": "HOLD", "close_price": None, "reason": "Ticker is required"})
    try:
        end = dt.datetime.now()
        start = end - pd.DateOffset(days=120)
        df = yf.download(ticker, start=start, end=end)
        if df is None or df.empty:
            return jsonify({"signal": "HOLD", "close_price": None, "reason": f"No data found for {ticker}"})
        from backtester_main import (
            RWB_daily_signal, ATR_Stop_daily_signal, RangeBreakout_daily_signal,
            Donchian_daily_signal, MACD_daily_signal, RSI_daily_signal, BollingerBand_daily_signal,
            ML_daily_signal
        )
        # Use a dictionary for strategy mapping
        def rangebreakout_fn(df):
            return RangeBreakout_daily_signal(df, int(longMA) if longMA else 50, int(shortMA) if shortMA else 20)
        def ml_fn(df):
            buy_confidence = float(request.args.get('buyConfidence', 0.5))
            sell_confidence = float(request.args.get('sellConfidence', 0.9))
            return ML_daily_signal(df, buy_confidence, sell_confidence)
        STRATEGY_MAP = {
            'RWB': RWB_daily_signal,
            'ATR_Stop': ATR_Stop_daily_signal,
            'RangeBreakout': rangebreakout_fn,
            'Donchian': Donchian_daily_signal,
            'MACD': MACD_daily_signal,
            'RSI': RSI_daily_signal,
            'BollingerBand': BollingerBand_daily_signal,
            'ML': ml_fn
        }
        result = STRATEGY_MAP.get(strategy, lambda df: {"signal": "HOLD", "close_price": None, "reason": f"Strategy {strategy} not supported for daily signals yet"})(df)
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        result = make_serializable(result)
        # Guarantee required fields in response (except reason for ML)
        required_fields = ["signal", "close_price"]
        if strategy != 'ML':
            required_fields.append("reason")
        
        for key in required_fields:
            if key not in result:
                if key == "signal":
                    result[key] = "HOLD"
                elif key == "close_price":
                    result[key] = None
                elif key == "reason":
                    result[key] = "No reason provided"
        return jsonify(result)
    except Exception as e:
        logging.error('Error in /api/daily_signal', exc_info=True)
        return jsonify({"signal": "HOLD", "close_price": None, "reason": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
