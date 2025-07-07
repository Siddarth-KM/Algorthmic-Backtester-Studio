import React, { useState, useEffect, useRef } from 'react';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

import { TrendingUp, Calendar, Target, Brain, Play, Download, AlertCircle, CheckCircle, Info } from 'lucide-react';

import ReactDOM from 'react-dom';

 

const BacktestingApp = () => {

  const [formData, setFormData] = useState({

    ticker: '',

    strategy: '',

    testPeriod: '',

    customStart: ''

  });

   
  
  const [results, setResults] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState('');
  
  const [resultsStrategy, setResultsStrategy] = useState(''); // Track which strategy was used for current results

   

  const [rangeBreakoutParams, setRangeBreakoutParams] = useState({

    longMA: '',

    shortMA: ''

  });

   
  
  const [openTooltip, setOpenTooltip] = useState(null);

  const [tooltipPos, setTooltipPos] = useState({ top: 0, left: 0 });

  const [mlButtonText, setMlButtonText] = useState(true); // true for first text, false for second

  const [showMlForm, setShowMlForm] = useState(false);

  const [mlFormData, setMlFormData] = useState({

    ticker: '',

    downloadStart: '3', // Default to 2010

    testPeriod: '4',    // Default to 1 year (updated for new numbering)

    buyConfidence: 0.5,

    sellConfidence: 0.9,

    customDownloadStart: '',

    customTestPeriod: ''

  });

  const iconRefs = useRef({});

  // Add state for daily signal
  const [dailySignal, setDailySignal] = useState(null);
  const [signalLoading, setSignalLoading] = useState(false);
  const [signalError, setSignalError] = useState(null);

 

  // Click-away handler for tooltip

  useEffect(() => {

    if (!openTooltip) return;

    function handleClick(e) {

      const ref = iconRefs.current[openTooltip];

      if (

        ref &&

        !ref.contains(e.target) &&

        document.getElementById('strategy-tooltip') &&

        !document.getElementById('strategy-tooltip').contains(e.target)

      ) {

        setOpenTooltip(null);

      }

    }

    document.addEventListener('mousedown', handleClick);

    return () => document.removeEventListener('mousedown', handleClick);

  }, [openTooltip]);

 

  // Your actual strategies based on backend

  const availableStrategies = [

    { id: '1', name: 'ATR Stop', description: 'ATR-based stop loss strategy' },

    { id: '2', name: 'RWB Strategy', description: 'Red White Blue momentum strategy' },

    { id: '3', name: 'Range Breakout', description: 'Moving average crossover breakout' },

    { id: '4', name: 'Donchian Channel', description: 'Donchian channel breakout strategy' },

    { id: '5', name: 'MACD', description: 'MACD signal strategy' },

    { id: '6', name: 'RSI', description: 'RSI oscillator strategy' },

    { id: '7', name: 'Bollinger Bands', description: 'Bollinger Bands strategy' }

  ].map(s => ({...s, key: s.id}));

 

  const testPeriods = [

    { id: '1', label: 'Last 4 days' },

    { id: '2', label: 'Last 1 week' },

    { id: '3', label: 'Last 2 weeks' },

    { id: '4', label: 'Last 1 month' },

    { id: '5', label: 'Last 3 months' },

    { id: '6', label: 'Last 6 months' },

    { id: '7', label: 'Custom Start Date' }

  ].map(p => ({...p, key: p.id}));

 

  // Combine similar event handlers for efficiency
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({...prev, [name]: type === 'checkbox' ? checked : value}));
  };

 

  const handleStrategySelect = (strategyId) => {
    setFormData(prev => ({
      ...prev,
      strategy: strategyId
    }));
    setResults(null);
    setDailySignal(null);
    setSignalError(null);
  };

 

  const handleRangeBreakoutChange = (e) => {
    const { name, value } = e.target;
    setRangeBreakoutParams(prev => ({...prev, [name]: value.replace(/[^0-9]/g, '')}));
  };

  const handleMlFormChange = (e) => {
    const { name, value, type } = e.target;
    setMlFormData(prev => ({...prev, [name]: type === 'number' ? parseFloat(value) : value}));
  };

  const handleMlButtonClick = () => {
    setMlButtonText(!mlButtonText);
    setShowMlForm(!showMlForm);
    // Reset all results and UI state when switching between ML and regular strategies
    setResults(null);
    setDailySignal(null);
    setSignalError(null);
    setResultsStrategy('');
    // Reset ML form data when switching to ML
    if (!showMlForm) {
      setMlFormData({
        ticker: '',
        downloadStart: '3',
        testPeriod: '4',
        buyConfidence: 0.5,
        sellConfidence: 0.9,
        customDownloadStart: '',
        customTestPeriod: ''
      });
    }
  }; 

 

  const getTradeLogHeight = (tradeCount) => {
    if (tradeCount <= 2) return 'h-56'; // ~224px
    return 'h-150'; // 3+ trades: max height (600px)
  };

  const handleSubmit = async () => {
    console.log('Starting backtest...');
    setLoading(true);
    setError('');
    try {
      let payload;
      if (showMlForm) {
        // ML Strategy payload
        payload = {
          strategy: '9',
          ticker: mlFormData.ticker,
          mlDownloadStart: mlFormData.downloadStart,
          mlTestPeriod: mlFormData.testPeriod,
          buyConfidence: mlFormData.buyConfidence,
          sellConfidence: mlFormData.sellConfidence
        };
      } else {
        // Standard strategy payload
        payload = {
        strategy: formData.strategy,
        ticker: formData.ticker,
        testPeriod: formData.testPeriod,
        customStart: formData.testPeriod === '7' ? formData.customStart : ''
      };
      if (formData.strategy === '3') {
        payload.longMA = rangeBreakoutParams.longMA || '50';
        payload.shortMA = rangeBreakoutParams.shortMA || '20';
      }
      }
      console.log('Sending payload:', payload);
      console.log('API URL:', '/api/backtest');
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      const data = await response.json();
      console.log('Response data:', data);
      if (data.error) {
        console.log('Error in response:', data.error);
        setError(data.error);
        setResults(null);
      } else {
        console.log('Setting results:', data);
        setResults(data);
        setResultsStrategy(showMlForm ? '9' : formData.strategy);
      }
    } catch (err) {
      console.error('Exception caught:', err);
      console.error('Error name:', err.name);
      console.error('Error message:', err.message);
      console.error('Error stack:', err.stack);
      setError(`Failed to run backtest: ${err.message}. Please check your inputs and try again.`);
    } finally {
      console.log('Setting loading to false');
      setLoading(false);
    }
  };

 

  const formatCurrency = (value) => {

    return new Intl.NumberFormat('en-US', {

      style: 'currency',

      currency: 'USD',

      minimumFractionDigits: 0

    }).format(value);

  };

 

  // Add a helper to get the selected strategy name

  const getStrategyName = () => {

    if (resultsStrategy === '9') {

      return 'ML Model';

    }

    return availableStrategies.find(s => s.id === resultsStrategy)?.name || 'Strategy';

  }; 
  // Helper to generate all dates between two dates (inclusive)
  function getAllDates(startDate, endDate) {
    const dates = [];
    let current = new Date(startDate);
    const end = new Date(endDate);
    while (current <= end) {
      dates.push(current.toISOString().split('T')[0]);
      current.setDate(current.getDate() + 1);
    }
    return dates;
  }

  // Calculate x-axis ticks for portfolio performance graph
  const performanceData = results && results.performance ? results.performance.map((v, i) => ({ x: i + 1, value: v })) : [{ x: 1, value: 0 }];
  const maxX = performanceData.length > 0 ? performanceData[performanceData.length - 1].x : 1;

  // Simple xTicks calculation like in the old working version
  let xTicks = [];
  if (maxX <= 8) {
    for (let i = 1; i <= maxX; i++) xTicks.push(i);
    } else {
    const numTicks = 8;
    for (let i = 0; i < numTicks; i++) {
      let tick = Math.floor(1 + (i * (maxX - 1) / (numTicks - 1)));
      xTicks.push(tick);
    }
    xTicks[0] = 1;
    xTicks[xTicks.length - 1] = maxX;
    xTicks = Array.from(new Set(xTicks)).sort((a, b) => a - b);
  }

 

  // Strategy description and optimal usage time frame for each strategy

  const strategyDescriptions = {

    '1': 'ATR-based stop loss strategy: Uses the Average True Range to dynamically set stop-loss levels, aiming to capture trends while limiting downside risk. Best for high volatility stocks. Best for multi-day to multi-week trends.',

    '2': 'The RWB (Red-White-Blue) strategy uses multiple short-term and long-term exponential moving averages. When all short-term averages are above all long-term averages, it signals a strong uptrend and a buy. When this condition ends, it signals an exit. This helps capture sustained trends. Best for moderate to high volatility stocks. Optimal for strong trends lasting several days to months.',

    '3': 'Moving average crossover breakout: Enters trades when a short-term moving average crosses above or below a long-term moving average, signaling a potential breakout. Best for high volatility stocks. Best for breakouts over a few days to a few weeks.',

    '4': 'Donchian channel breakout strategy: Buys when price breaks above the highest high of the last N days and sells when it falls below the lowest low. Best for high volatility stocks. Works well for moves spanning several days to weeks.',

    '5': 'MACD signal strategy: Uses the Moving Average Convergence Divergence indicator to identify bullish and bearish momentum shifts. Best for moderate to high volatility stocks. Best for swings lasting days to weeks.',

    '6': 'RSI oscillator strategy: Trades based on the Relative Strength Index, buying when oversold and selling when overbought. Works in both low and high volatility stocks. Optimal for reversals over several days or more.',

    '7': 'Bollinger Bands strategy: Uses price volatility bands to identify overbought/oversold conditions and potential reversal points. Best for high volatility stocks. Best for volatility cycles of a few days to a few weeks.'

  }; 

 

  // Helper to fetch daily signal for the selected strategy
  const fetchDailySignal = async () => {
    setSignalLoading(true);
    setSignalError(null);
    setDailySignal(null);
    let endpoint = '';
    let params = '';
    // Map strategy id to endpoint
    switch (resultsStrategy) {
      case '1': endpoint = '/api/daily-signal/ATR_Stop'; break;
      case '2': endpoint = '/api/daily-signal/RWB'; break;
      case '3': endpoint = '/api/daily-signal/RangeBreakout';
        // Add params for Range Breakout if needed
        if (formData.rangeBreakoutLong && formData.rangeBreakoutShort) {
          params = `?longMA=${formData.rangeBreakoutLong}&shortMA=${formData.rangeBreakoutShort}`;
        }
        break;
      case '4': endpoint = '/api/daily-signal/Donchian'; break;
      case '5': endpoint = '/api/daily-signal/MACD'; break;
      case '6': endpoint = '/api/daily-signal/RSI'; break;
      case '7': endpoint = '/api/daily-signal/BollingerBand'; break;
      case '9': endpoint = '/api/daily-signal/ML';
        params = `?buyConfidence=${mlFormData.buyConfidence}&sellConfidence=${mlFormData.sellConfidence}`;
        break;
      default: setSignalError('No daily signal available for this strategy.'); setSignalLoading(false); return;
    }
    try {
      const res = await fetch(`${endpoint}?ticker=${showMlForm ? mlFormData.ticker : formData.ticker}${params}`);
      if (!res.ok) throw new Error('Failed to fetch daily signal');
      const data = await res.json();
      setDailySignal(data);
    } catch (err) {
      setSignalError('Failed to fetch daily signal.');
    } finally {
      setSignalLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header Section */}
      <div className="container mx-auto px-4 pt-8 pb-4">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            Algorithmic Backtester Studio
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Bring institutional-grade quantitative strategies to your portfolio with easy-to-use backtesting and daily trading signals
          </p>
        </div>
      </div>
      
      <div className="container mx-auto px-4 pb-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 min-h-[80vh] items-start">
          {/* Input Form */}
          {showMlForm ? (
          <div className="lg:col-span-1 flex flex-col justify-start">
              {/* ML Configuration Form */}
              <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                  <Info 
                    className="text-blue-400 cursor-pointer" 
                    onClick={(e) => {
                      e.stopPropagation();
                      const rect = e.currentTarget.getBoundingClientRect();
                      setTooltipPos({
                        top: rect.bottom + window.scrollY + 8,
                        left: rect.left + window.scrollX + rect.width / 2
                      });
                      setOpenTooltip(openTooltip === 'ml-info' ? null : 'ml-info');
                    }}
                    tabIndex={0}
                    aria-label="ML Model Info"
                  />
                  <span className="ml-3">ML Model Configuration</span>
                  {openTooltip === 'ml-info' && ReactDOM.createPortal(
                    <div
                      id="strategy-tooltip"
                      className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                      style={{
                        position: 'absolute',
                        top: tooltipPos.top,
                        left: tooltipPos.left,
                        transform: 'translate(-50%, 0)',
                      }}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-bold text-blue-300">ML Model</span>
                        <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                          &times;
                        </button>
                      </div>
                      <div>
                        This ML model uses XGBoost to predict stock price movements based on 25+ technical indicators including RSI, MACD, Bollinger Bands, and volume metrics. It trains on historical data and predicts buy/sell signals with confidence thresholds. The model identifies patterns that historically led to profitable trades, aiming to outperform traditional buy-and-hold strategies.
                      </div>
                    </div>,
                    document.body
                  )}
                </h2>
              <div className="space-y-6">
                {/* Stock Ticker */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Stock Ticker
                  </label>
                  <input
                    type="text"
                    name="ticker"
                    value={mlFormData.ticker}
                    onChange={handleMlFormChange}
                    placeholder="e.g., AAPL, TSLA, SPY"
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    required
                  />
                </div>

                {/* Download Start Date */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center">
                    Download Start Date
                    <div className="relative group ml-2">
                      <Info 
                        className="w-4 h-4 text-blue-400 cursor-pointer" 
                        onClick={(e) => {
                          e.stopPropagation();
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'download-start' ? null : 'download-start');
                        }}
                        tabIndex={0}
                        aria-label="Download Start Date Info"
                      />
                      {openTooltip === 'download-start' && ReactDOM.createPortal(
                        <div
                          id="strategy-tooltip"
                          className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{
                            position: 'absolute',
                            top: tooltipPos.top,
                            left: tooltipPos.left,
                            transform: 'translate(-50%, 0)',
                          }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Download Start Date</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                              &times;
                            </button>
                          </div>
                          <div>
                            The starting year for downloading historical data to train the ML model. Earlier dates provide more training data but may include outdated market patterns.
                          </div>
                        </div>,
                        document.body
                      )}
                    </div>
                  </label>
                  <select
                    name="downloadStart"
                    value={mlFormData.downloadStart}
                    onChange={handleMlFormChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    required
                  >
                    <option value="1" className="text-black">2000</option>
                    <option value="2" className="text-black">2005</option>
                    <option value="3" className="text-black">2010</option>
                    <option value="4" className="text-black">2015</option>
                    <option value="custom" className="text-black">Custom Date</option>
                  </select>
                  {mlFormData.downloadStart === 'custom' && (
                    <input
                      type="date"
                      name="customDownloadStart"
                      value={mlFormData.customDownloadStart}
                      onChange={handleMlFormChange}
                      className="w-full mt-2 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                      required
                    />
                  )}
                </div>

                {/* Test Period */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Test Period
                  </label>
                  <select
                    name="testPeriod"
                    value={mlFormData.testPeriod}
                    onChange={handleMlFormChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    required
                  >
                    <option value="1" className="text-black">Last 3 months</option>
                    <option value="2" className="text-black">Last 6 months</option>
                    <option value="3" className="text-black">YTD (Year to Date)</option>
                    <option value="4" className="text-black">Last 1 year</option>
                    <option value="5" className="text-black">Last 2 years</option>
                    <option value="custom" className="text-black">Custom Date</option>
                  </select>
                  {mlFormData.testPeriod === 'custom' && (
                    <input
                      type="date"
                      name="customTestPeriod"
                      value={mlFormData.customTestPeriod}
                      onChange={handleMlFormChange}
                      className="w-full mt-2 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                      required
                    />
                  )}
                </div>

                {/* Buy Confidence */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center">
                    Buy Confidence Threshold
                    <div className="relative group ml-2">
                      <Info 
                        className="w-4 h-4 text-blue-400 cursor-pointer" 
                        onClick={(e) => {
                          e.stopPropagation();
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'buy-confidence' ? null : 'buy-confidence');
                        }}
                        tabIndex={0}
                        aria-label="Buy Confidence Info"
                      />
                      {openTooltip === 'buy-confidence' && ReactDOM.createPortal(
                        <div
                          id="strategy-tooltip"
                          className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{
                            position: 'absolute',
                            top: tooltipPos.top,
                            left: tooltipPos.left,
                            transform: 'translate(-50%, 0)',
                          }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Buy Confidence</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                              &times;
                            </button>
                          </div>
                          <div>
                            The minimum confidence level (0.1-1.0) required for the ML model to generate a BUY signal. Higher values mean more conservative buying - only signals with high confidence will trigger buys. Lower values mean more aggressive buying.
                          </div>
                        </div>,
                        document.body
                      )}
                    </div>
                  </label>
                  <input
                    type="number"
                    name="buyConfidence"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={mlFormData.buyConfidence}
                    onChange={handleMlFormChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    placeholder="0.5"
                    defaultValue="0.5"
                  />
                </div>

                {/* Sell Confidence */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2 flex items-center">
                    Sell Confidence Threshold
                    <div className="relative group ml-2">
                      <Info 
                        className="w-4 h-4 text-blue-400 cursor-pointer" 
                        onClick={(e) => {
                          e.stopPropagation();
                          const rect = e.currentTarget.getBoundingClientRect();
                          setTooltipPos({
                            top: rect.bottom + window.scrollY + 8,
                            left: rect.left + window.scrollX + rect.width / 2
                          });
                          setOpenTooltip(openTooltip === 'sell-confidence' ? null : 'sell-confidence');
                        }}
                        tabIndex={0}
                        aria-label="Sell Confidence Info"
                      />
                      {openTooltip === 'sell-confidence' && ReactDOM.createPortal(
                        <div
                          id="strategy-tooltip"
                          className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                          style={{
                            position: 'absolute',
                            top: tooltipPos.top,
                            left: tooltipPos.left,
                            transform: 'translate(-50%, 0)',
                          }}
                        >
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-bold text-blue-300">Sell Confidence</span>
                            <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                              &times;
                            </button>
                          </div>
                          <div>
                            The minimum confidence level (0.1-1.0) required for the ML model to generate a SELL signal. Higher values mean more conservative selling - only signals with high confidence will trigger sells. Lower values mean more aggressive selling.
                          </div>
                        </div>,
                        document.body
                      )}
                    </div>
                  </label>
                  <input
                    type="number"
                    name="sellConfidence"
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    value={mlFormData.sellConfidence}
                    onChange={handleMlFormChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    placeholder="0.9"
                    defaultValue="0.9"
                  />
                </div>

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={loading || !mlFormData.ticker}
                  className="w-full bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 disabled:scale-100 flex items-center justify-center"
                >
                  {loading ? (
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                  ) : (
                    <Brain className="mr-3" />
                  )}
                  {loading ? 'Running ML Backtest...' : 'Start ML Backtest'}
                </button>

                {/* Daily Signal Button - ML */}
                {results && resultsStrategy === '9' && (
                  <div className="w-full flex flex-col items-center mt-3">
                    <button
                      onClick={fetchDailySignal}
                      className="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all duration-300 flex items-center justify-center mb-3"
                      style={{ minHeight: '56px' }}
                      disabled={signalLoading}
                    >
                      {signalLoading ? (
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                      ) : (
                        <Target className="mr-3" />
                      )}
                      {signalLoading ? 'Loading Signal...' : "See Today's Signal"}
                    </button>
                    {signalError && (
                      <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 text-white w-full max-w-md text-center mt-2">{signalError}</div>
                    )}
                    {dailySignal && (
                      <div className="bg-white/10 border border-green-400 rounded-xl p-6 w-full max-w-md text-center mt-2">
                        <div className="text-lg font-bold mb-2">
                          <span className="text-white">Today's Signal:</span> {dailySignal.signal === 'BUY' && <span className="text-green-400">BUY</span>}
                          {dailySignal.signal === 'SELL' && <span className="text-red-400">SELL</span>}
                          {dailySignal.signal === 'HOLD' && <span className="text-yellow-300">HOLD</span>}
                        </div>
                        <div className="text-white mb-1">
                          Close Price: {typeof dailySignal.close_price === 'number' && !isNaN(dailySignal.close_price) ? `$${dailySignal.close_price.toFixed(2)}` : 'N/A'}
                        </div>
                            {resultsStrategy !== '9' && dailySignal.reason && dailySignal.reason !== 'undefined' && (
      <div className="text-gray-300 text-sm">
        {dailySignal.reason && dailySignal.reason.startsWith('Reason:') ? dailySignal.reason : `Reason: ${dailySignal.reason}`}
      </div>
    )}
    {/* Show confidence for non-ML strategies */}
    {resultsStrategy !== '9' && dailySignal.confidence && (
      <div className="text-blue-300 text-xs mt-1">
        Confidence: {(dailySignal.confidence * 100).toFixed(1)}%
      </div>
    )}
    {/* Show all 3 probabilities for ML daily signal */}
    {resultsStrategy === '9' && dailySignal.all_probabilities && (
      <div className="text-sm text-white mt-2">
        <div className="font-semibold text-green-300 mb-1">Probabilities:</div>
        {dailySignal.all_probabilities.map((prob, idx) => (
          <div key={idx} className="flex justify-center gap-2">
            <span className="font-mono text-gray-200">{prob.class}:</span>
            <span className="font-mono text-blue-200">{(prob.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    )}
                      </div>
                    )}
                  </div>
                )}

                {error && (
                  <div className="mt-4 p-4 bg-red-500/20 border border-red-500/30 rounded-xl flex items-center">
                    <AlertCircle className="mr-3 text-red-400" />
                    <p className="text-red-200">{error}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
          ) : (
            <div className="lg:col-span-1 flex flex-col justify-start">
              {/* Standard Strategy Configuration Form */}
            <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
              <h2 className="text-2xl font-bold text-white mb-6 flex items-center">
                <Target className="mr-3 text-blue-400" />
                Strategy & Backtest Configuration
               </h2>
              
              <div className="space-y-6">
                {/* Stock Ticker */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Stock Ticker
                  </label>
                  <input
                    type="text"
                    name="ticker"
                    value={formData.ticker}
                    onChange={handleInputChange}
                    placeholder="e.g., AAPL, TSLA, SPY"
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    required
                  />
                </div>

                {/* Test Period */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Test Period
                  </label>
                  <select
                    name="testPeriod"
                    value={formData.testPeriod}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                    required
                  >
                    <option value="" className="text-black">Select period</option>
                    {testPeriods.map(period => (
                        <option key={period.id} value={period.id} className="text-black">
                          {period.label}
                        </option>
                    ))}
                  </select>
                    {formData.testPeriod === '7' && (
                      <input
                        type="date"
                        name="customStart"
                        value={formData.customStart}
                        onChange={handleInputChange}
                        className="w-full mt-2 px-4 py-3 bg-white/5 border border-white/20 rounded-xl text-white focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all"
                        required
                      />
                    )}
                </div>

                {/* Strategy Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-4">
                    Select Strategy
                  </label>
                  <div className="space-y-3">
                    {availableStrategies.map(strategy => (
                      <div
                        key={strategy.id}
                        className={`p-4 rounded-xl border-2 transition-all cursor-pointer ${
                          formData.strategy === strategy.id
                            ? 'border-blue-400 bg-blue-400/10'
                            : 'border-white/20 bg-white/5 hover:border-white/40'
                        }`}
                        onClick={() => handleStrategySelect(strategy.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="font-medium text-white">{strategy.name}</h4>
                            <p className="text-sm text-gray-400">{strategy.description}</p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <div
                              className="relative group"
                              ref={el => (iconRefs.current[strategy.id] = el)}
                              onClick={e => e.stopPropagation()}
                            >
                              <Info
                                className="w-5 h-5 text-blue-400 cursor-pointer"
                                onClick={e => {
                                  e.stopPropagation();
                                    const rect = e.currentTarget.getBoundingClientRect();
                                    setTooltipPos({
                                      top: rect.bottom + window.scrollY + 8,
                                      left: rect.left + window.scrollX + rect.width / 2
                                    });
                                  setOpenTooltip(openTooltip === strategy.id ? null : strategy.id);
                                }}
                                tabIndex={0}
                                aria-label={`${strategy.name} Info`}
                              />
                              {openTooltip === strategy.id && ReactDOM.createPortal(
                                <div
                                  id="strategy-tooltip"
                                  className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                  style={{
                                    position: 'absolute',
                                    top: tooltipPos.top,
                                    left: tooltipPos.left,
                                    transform: 'translate(-50%, 0)',
                                  }}
                                >
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-blue-300">{strategy.name}</span>
                                      <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                                        &times;
                                      </button>
                                  </div>
                                  <div>
                                    {strategyDescriptions[strategy.id]}
                                  </div>
                                </div>,
                                document.body
                              )}
                            </div>
                              <div
                                className={`w-5 h-5 rounded-full border-2 ${
                              formData.strategy === strategy.id
                                ? 'border-blue-400 bg-blue-400'
                                : 'border-gray-400'
                                }`}
                              >
                              {formData.strategy === strategy.id && (
                                <CheckCircle className="w-5 h-5 text-white" />
                              )}
                            </div>
                          </div>
                        </div>
                          
                          {/* Range Breakout parameters - show only when selected */}
                          {formData.strategy === strategy.id && strategy.id === '3' && (
                            <div className="mt-4 pt-4 border-t border-white/20" onClick={e => e.stopPropagation()}>
                              <div className="flex flex-col md:flex-row gap-4">
                        <div className="flex-1">
                                  <label className="block text-sm font-medium text-gray-300 mb-2">
                                    Long MA Days
                                  </label>
                          <input
                            type="number"
                            name="longMA"
                            min="1"
                            value={rangeBreakoutParams.longMA}
                            onChange={handleRangeBreakoutChange}
                                    className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all text-sm"
                            placeholder="e.g., 50"
                          />
                        </div>
                        <div className="flex-1">
                                  <label className="block text-sm font-medium text-gray-300 mb-2">
                                    Short MA Days
                                  </label>
                          <input
                            type="number"
                            name="shortMA"
                            min="1"
                            value={rangeBreakoutParams.shortMA}
                            onChange={handleRangeBreakoutChange}
                                    className="w-full px-3 py-2 bg-white/5 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-400/20 transition-all text-sm"
                            placeholder="e.g., 20"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                        </div>
                      ))}
                    </div>
                </div>

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={loading || !formData.ticker || !formData.testPeriod || !formData.strategy}
                  className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 disabled:scale-100 flex items-center justify-center"
                >
                  {loading ? (
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                  ) : (
                    <Play className="mr-3" />
                  )}
                  {loading ? 'Running Backtest...' : 'Start Backtest'}
                </button>

                  {/* Daily Signal Button - Regular Strategies */}
                  {results && resultsStrategy !== '9' && (
                    <div className="w-full flex flex-col items-center mt-3">
                      <button
                        onClick={fetchDailySignal}
                        className="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all duration-300 flex items-center justify-center mb-3"
                        style={{ minHeight: '56px' }}
                        disabled={signalLoading}
                      >
                        {signalLoading ? (
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                        ) : (
                          <Target className="mr-3" />
                        )}
                        {signalLoading ? 'Loading Signal...' : "See Today's Signal"}
                      </button>
                      {signalError && (
                        <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4 text-white w-full max-w-md text-center mt-2">{signalError}</div>
                      )}
                      {dailySignal && (
                        <div className="bg-white/10 border border-green-400 rounded-xl p-6 w-full max-w-md text-center mt-2">
                          <div className="text-lg font-bold mb-2">
                            <span className="text-white">Today's Signal:</span> {dailySignal.signal === 'BUY' && <span className="text-green-400">BUY</span>}
                            {dailySignal.signal === 'SELL' && <span className="text-red-400">SELL</span>}
                            {dailySignal.signal === 'HOLD' && <span className="text-yellow-300">HOLD</span>}
                          </div>
                          <div className="text-white mb-1">
                            Close Price: {typeof dailySignal.close_price === 'number' && !isNaN(dailySignal.close_price) ? `$${dailySignal.close_price.toFixed(2)}` : 'N/A'}
                          </div>
                          {resultsStrategy !== '9' && dailySignal.reason && dailySignal.reason !== 'undefined' && (
                            <div className="text-gray-300 text-sm">
                              {dailySignal.reason && dailySignal.reason.startsWith('Reason:') ? dailySignal.reason : `Reason: ${dailySignal.reason}`}
                            </div>
                          )}
                          {dailySignal.confidence && (
                            <div className="text-blue-300 text-xs mt-1">
                              Confidence: {(dailySignal.confidence * 100).toFixed(1)}%
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                {error && (
                  <div className="mt-4 p-4 bg-red-500/20 border border-red-500/30 rounded-xl flex items-center">
                    <AlertCircle className="mr-3 text-red-400" />
                    <p className="text-red-200">{error}</p>
                  </div>
                )}
                  </div>
              </div>
            </div>
          )}

          {/* Results and Graph Area */}
          <div className="lg:col-span-2 flex flex-col justify-end">
            {/* ML Model Option */}
            <div className="w-full mb-8 flex items-start">
              <button
                className="w-full py-8 bg-gradient-to-r from-emerald-600 to-teal-700 text-white text-2xl font-bold rounded-3xl shadow-lg border-4 border-white/20 hover:from-emerald-700 hover:to-teal-800 transition-all duration-300 cursor-pointer self-start"
                style={{ letterSpacing: '0.02em' }}
                onClick={handleMlButtonClick}
              >
                {mlButtonText
                  ? "Optimize returns with a Machine Learning Model"
                  : "Optimize returns using Top Quantitative Strategies"
                }
              </button>
            </div>

            {/* Stats and Trade Log Row - now stacked, stats box on top, trade log below */}
            <div className="flex flex-col gap-6 mb-8 min-h-0">
              {/* Strategy Statistics Box - now on top */}
              {results && (results.stats || results.totalReturn !== undefined) && (
                <div className="w-full bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 h-66 flex flex-col min-h-0 justify-start min-w-0 order-1 relative">
                  <div className="flex items-center mb-2 justify-center">
                    <h3 className="text-2xl font-bold text-white">Strategy Statistics</h3>
                        </div>

                  <div className="overflow-y-auto flex-1 min-h-0">
                  {results.stats ? (
                    <pre className="text-white font-mono text-lg whitespace-pre-line leading-relaxed w-full max-w-none">
                      {results.stats}
                    </pre>
                  ) : (
                    <div className="text-white font-mono text-base space-y-2 w-full max-w-none">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        <div className="whitespace-nowrap">{getStrategyName()} Return: {results.totalReturn}%</div>
                      <div className="whitespace-nowrap">Buy & Hold: {results.buyHoldReturn}%</div>
                      <div className="whitespace-nowrap">Win Rate: {results.winRate}%</div>
                      <div className="whitespace-nowrap">Total Trades: {results.trades}</div>
                      <div className="whitespace-nowrap">Avg Return/Trade: {results.avgReturnPerTrade}%</div>
                        <div className="whitespace-nowrap">
                          Sharpe Ratio: {results.sharpeRatio || 'N/A'}
                          <div className="relative ml-2 inline-block">
                            <Info 
                              className="w-4 h-4 text-blue-400 cursor-pointer" 
                              onClick={(e) => {
                                e.stopPropagation();
                                const rect = e.currentTarget.getBoundingClientRect();
                                setTooltipPos({
                                  top: rect.bottom + window.scrollY + 8,
                                  left: rect.left + window.scrollX + rect.width / 2
                                });
                                setOpenTooltip(openTooltip === 'sharpe' ? null : 'sharpe');
                              }}
                              tabIndex={0}
                              aria-label="Sharpe Ratio Info"
                            />
                            {openTooltip === 'sharpe' && ReactDOM.createPortal(
                              <div
                                id="strategy-tooltip"
                                className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                style={{
                                  position: 'absolute',
                                  top: tooltipPos.top,
                                  left: tooltipPos.left,
                                  transform: 'translate(-50%, 0)',
                                }}
                              >
                                <div className="flex justify-between items-center mb-1">
                                  <span className="font-bold text-blue-300">Sharpe Ratio</span>
                                  <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                                    &times;
                                  </button>
                    </div>
                                <div>
                                  Sharpe Ratio shows how much return you get for the risk you take. Higher numbers mean better performance for the amount of risk. Above 1.0 is good.
                                </div>
                              </div>,
                              document.body
                  )}
                </div>
                        </div>
                        {results.calmarRatio !== undefined && (
                          <div className="whitespace-nowrap">
                            Calmar Ratio: {results.calmarRatio || 'N/A'}
                            <div className="relative ml-2 inline-block">
                              <Info 
                                className="w-4 h-4 text-blue-400 cursor-pointer" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const rect = e.currentTarget.getBoundingClientRect();
                                  setTooltipPos({
                                    top: rect.bottom + window.scrollY + 8,
                                    left: rect.left + window.scrollX + rect.width / 2
                                  });
                                  setOpenTooltip(openTooltip === 'calmar' ? null : 'calmar');
                                }}
                                tabIndex={0}
                                aria-label="Calmar Ratio Info"
                              />
                              {openTooltip === 'calmar' && ReactDOM.createPortal(
                                <div
                                  id="strategy-tooltip"
                                  className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                  style={{
                                    position: 'absolute',
                                    top: tooltipPos.top,
                                    left: tooltipPos.left,
                                    transform: 'translate(-50%, 0)',
                                  }}
                                >
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-blue-300">Calmar Ratio</span>
                                    <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                                      &times;
                                    </button>
                                  </div>
                                  <div>
                                    Calmar Ratio compares yearly returns to the biggest loss. Higher numbers mean you get more return for the worst loss you might face.
                                    <br /><br />
                                    <span className="text-yellow-300">Note: Requires at least 2 trades for accurate calculation. Shows N/A with fewer trades.</span>
                                  </div>
                                </div>,
                                document.body
                              )}
                            </div>
                          </div>
                        )}
                        {results.sortinoRatio !== undefined && (
                          <div className="whitespace-nowrap">
                            Sortino Ratio: {results.sortinoRatio || 'N/A'}
                            <div className="relative ml-2 inline-block">
                              <Info 
                                className="w-4 h-4 text-blue-400 cursor-pointer" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const rect = e.currentTarget.getBoundingClientRect();
                                  setTooltipPos({
                                    top: rect.bottom + window.scrollY + 8,
                                    left: rect.left + window.scrollX + rect.width / 2
                                  });
                                  setOpenTooltip(openTooltip === 'sortino' ? null : 'sortino');
                                }}
                                tabIndex={0}
                                aria-label="Sortino Ratio Info"
                              />
                              {openTooltip === 'sortino' && ReactDOM.createPortal(
                                <div
                                  id="strategy-tooltip"
                                  className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                  style={{
                                    position: 'absolute',
                                    top: tooltipPos.top,
                                    left: tooltipPos.left,
                                    transform: 'translate(-50%, 0)',
                                  }}
                                >
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-blue-300">Sortino Ratio</span>
                                    <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                                      &times;
                                    </button>
                                  </div>
                                  <div>
                                    Sortino Ratio looks at returns compared to only the bad days (losses). Higher numbers mean better performance when things go wrong.
                                    <br /><br />
                                    <span className="text-yellow-300">Note: Requires at least 2 trades for accurate calculation. Shows N/A with fewer trades.</span>
                                  </div>
                                </div>,
                                document.body
                              )}
                            </div>
                          </div>
                        )}
                        {results.profitFactor !== undefined && (
                          <div className="whitespace-nowrap">
                            Profit Factor: {results.profitFactor || 'N/A'}
                            <div className="relative ml-2 inline-block">
                              <Info 
                                className="w-4 h-4 text-blue-400 cursor-pointer" 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  const rect = e.currentTarget.getBoundingClientRect();
                                  setTooltipPos({
                                    top: rect.bottom + window.scrollY + 8,
                                    left: rect.left + window.scrollX + rect.width / 2
                                  });
                                  setOpenTooltip(openTooltip === 'profit' ? null : 'profit');
                                }}
                                tabIndex={0}
                                aria-label="Profit Factor Info"
                              />
                              {openTooltip === 'profit' && ReactDOM.createPortal(
                                <div
                                  id="strategy-tooltip"
                                  className="z-[9999] w-64 p-3 bg-gray-800 text-white text-xs rounded shadow-lg border border-blue-400"
                                  style={{
                                    position: 'absolute',
                                    top: tooltipPos.top,
                                    left: tooltipPos.left,
                                    transform: 'translate(-50%, 0)',
                                  }}
                                >
                                  <div className="flex justify-between items-center mb-1">
                                    <span className="font-bold text-blue-300">Profit Factor</span>
                                    <button className="text-white text-lg ml-2" onClick={() => setOpenTooltip(null)}>
                                      &times;
                                    </button>
                                  </div>
                                  <div>
                                    Profit Factor compares total wins to total losses. Above 1.5 means you're winning more than losing. Above 2.0 is really good.
                                  </div>
                                </div>,
                                document.body
                              )}
                            </div>
                          </div>
                        )}
                        {results.avgDaysInTrade !== undefined && <div className="whitespace-nowrap">Avg Days in Trade: {results.avgDaysInTrade || 'N/A'}</div>}
                      </div>
                    </div>
                  )}
                  </div>
                </div>
              )}



              {/* Trade Log - now below */}
              {results && results.entry_dates && results.exit_dates && results.returns && (
                <div className={`w-full bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 ${getTradeLogHeight(results.entry_dates.length)} flex flex-col min-h-0 justify-end min-w-0 order-2 mt-0`}>
                  <h3 className="text-2xl font-bold text-white mb-6 text-center">Trade Log</h3>
                  <div className={`${results.entry_dates.length > 10 ? 'overflow-y-auto' : 'overflow-y-visible'} flex-1 min-h-0`}>
                    <table className="w-full text-white">
                      <thead>
                        <tr>
                          <th className="px-2 py-1 text-base text-center">Entry Date</th>
                          <th className="px-2 py-1 text-base text-center">Exit Date</th>
                          <th className="px-2 py-1 text-base text-center">Return</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.entry_dates.map((entry, idx) => {
                          const entryDate = new Date(entry);
                          const exitDate = new Date(results.exit_dates[idx]);
                          const entryFormatted = `${(entryDate.getMonth() + 1).toString().padStart(2, '0')}/${entryDate.getDate().toString().padStart(2, '0')}/${entryDate.getFullYear().toString().slice(-2)}`;
                          const exitFormatted = `${(exitDate.getMonth() + 1).toString().padStart(2, '0')}/${exitDate.getDate().toString().padStart(2, '0')}/${exitDate.getFullYear().toString().slice(-2)}`;
                          return (
                          <tr key={idx}>
                              <td className="px-2 py-1 text-base text-center">{entryFormatted}</td>
                              <td className="px-2 py-1 text-base text-center">{exitFormatted}</td>
                              <td className="px-2 py-1 text-base text-center">{(results.returns[idx] * 100).toFixed(2)}%</td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>

            {/* Matplotlib Plot from Backend - now in the middle */}
            {results && results.plot_image && (
              <div className="w-full mb-8 bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20">
                <h3 className="text-xl font-bold text-white mb-6 text-center">Strategy Analysis Chart</h3>
                <div className="flex justify-center">
                  <img 
                    src={`data:image/png;base64,${results.plot_image}`} 
                    alt="Strategy Analysis" 
                    className="max-w-full h-auto rounded-lg shadow-lg"
                    style={{ maxHeight: '500px' }}
                  />
                </div>
              </div>
            )}

            {/* Portfolio Performance Graph - now at the bottom */}
            {results && results.performance && results.performance.length > 0 && (
            <div className="w-full">
              <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 pb-12 border border-white/20 h-96 flex flex-col justify-end min-w-0">
                  <h3 className="text-xl font-bold text-white mb-6 text-center">Portfolio Value</h3>
                <div className="flex-1 min-w-0">
                    <div className="relative w-full h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="x" 
                        stroke="#9CA3AF" 
                        label={{ value: 'Days', position: 'insideBottom', offset: -1 }} 
                        ticks={xTicks}
                        domain={['dataMin', 'dataMax']}
                        scale="point"
                      />
                      <YAxis stroke="#9CA3AF" />
                          <Tooltip
                            labelStyle={{ color: '#1F2937' }}
                            contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }}
                          />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#3B82F6"
                            strokeWidth={3}
                            name="Cumulative Return"
                            dot={false}
                          />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
              </div>
            )}
            


            {/* Error or Ready Message */}
            {results && results.error ? (
              <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-6 text-white mt-6">
                <AlertCircle className="mr-3 text-red-400 inline-block" />
                {results.error}
              </div>
            ) : !results && (
              <div className="flex flex-col justify-center items-center h-full">
                <TrendingUp className="mx-auto mb-6 text-gray-400" size={64} />
                <h3 className="text-2xl font-bold text-white mb-4">Ready to Backtest</h3>
                <p className="text-gray-300 max-w-md mx-auto">
                  Configure your strategy parameters and run a backtest to see detailed performance analytics and insights.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

 

export default BacktestingApp;