# Algorithmic Backtester Studio

A comprehensive web application that combines traditional trading strategies with machine learning models for backtesting and performance analysis of common stocks. Features a modern React frontend with real-time data visualization and a robust Flask backend with advanced statistical analysis.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
# On Windows:
py start_app.py

# On macOS/Linux:
python start_app.py
```

### Option 2: Manual Setup

#### Backend Setup
1. Install Python dependencies:
```bash
# On Windows:
py -m pip install -r requirements.txt

# On macOS/Linux:
pip install -r requirements.txt
```

2. Start the backend API server:
```bash
# On Windows:
py backend_api.py

# On macOS/Linux:
python backend_api.py
```

The backend will be available at `http://localhost:5000`

#### Frontend Setup
1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
Projects/
├── backend_api.py          # Flask API server with error logging
├── backtester_main.py      # Core backtesting logic for all strategies
├── ML_Project.py          # Machine learning backtesting implementation
├── Data_Analyzer.py       # Data analysis and visualization utilities
├── Financials_Viewer.py   # Financial data viewing components
├── backup.py              # Backup and utility functions
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── App.js         # Main React component with dynamic UI
│   │   ├── index.js       # React entry point
│   │   └── index.css      # Global styles
│   ├── public/            # Static assets
│   └── package.json       # Frontend dependencies
├── tests/                 # Test scripts and validation
│   ├── test_api.py        # API endpoint testing
│   ├── test_bollinger.py  # Bollinger Bands strategy testing
│   ├── test_connection.py # Connection and data validation
│   └── Test_1.py          # General functionality testing
├── stubs/                 # Type hints and stubs
├── requirements.txt       # Python dependencies
├── package.json           # Root package configuration
├── pyproject.toml         # Python project configuration
├── start_app.py          # Automated startup script
├── start_app.bat         # Windows batch startup script
└── README.md             # This file
```

## 🔧 Features

### Available Trading Strategies
- **RWB Strategy**: Red White Blue momentum strategy using multiple EMAs
- **SMA Strategy**: Simple Moving Average crossover strategy
- **Range Breakout**: Range breakout trading strategy
- **Donchian Channel**: Donchian channel breakout strategy
- **MACD**: MACD signal crossover strategy
- **RSI**: RSI oscillator overbought/oversold strategy
- **Bollinger Bands**: Bollinger Bands mean reversion strategy
- **ATR Stop**: ATR-based stop loss strategy
- **ML Model**: Machine Learning prediction model with confidence thresholds

### Advanced Performance Metrics
All strategies now include comprehensive performance analysis:
- **Total Return**: Overall strategy performance
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return measure
- **Calmar Ratio**: Return vs maximum drawdown
- **Sortino Ratio**: Downside risk-adjusted return
- **Profit Factor**: Gross profit vs gross loss
- **Average Days in Trade**: Average holding period
- **Buy & Hold Comparison**: Strategy vs market performance

### Testing Periods
- **Quick Testing**: Last 1-6 months
- **Custom Date Range**: User-defined start and end dates
- **ML-Specific Periods**: Separate training and testing periods for machine learning

### User Interface Features
- **Dynamic Trade Log**: Automatically adjusts height based on number of trades
- **Interactive Tooltips**: Detailed explanations for complex metrics
- **Real-time Charts**: Price action with trade entry/exit markers
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: Clear error messages and validation

## 🌐 API Endpoints

### POST /api/backtest
Runs a comprehensive backtest with the specified parameters.

**Standard Strategy Request:**
```json
{
  "strategy": "1",
  "ticker": "AAPL",
  "testPeriod": "3",
  "customStart": "2023-01-01"
}
```

**ML Strategy Request:**
```json
{
  "strategy": "9",
  "ticker": "AAPL",
  "mlDownloadStart": "3",
  "mlTestPeriod": "2",
  "buyConfidence": 0.7,
  "sellConfidence": 0.6
}
```

**Response:**
```json
{
  "totalReturn": 15.5,
  "winRate": 65.2,
  "trades": 12,
  "sharpeRatio": 1.8,
  "calmarRatio": 2.1,
  "sortinoRatio": 2.3,
  "profitFactor": 1.9,
  "avgDaysInTrade": 5.2,
  "buyHoldReturn": 8.3,
  "avgReturnPerTrade": 1.3,
  "plot_image": "base64_encoded_chart",
  "entry_dates": ["2023-01-15", "2023-02-01"],
  "exit_dates": ["2023-01-20", "2023-02-05"],
  "returns": [0.05, -0.02]
}
```

### GET /api/daily-signal/<strategy>
Returns current trading signal for the specified strategy.

## 🛠️ Dependencies

### Backend (Python)
- **Flask >= 2.0.0**: Web framework
- **Flask-CORS >= 3.0.0**: Cross-origin resource sharing
- **pandas >= 1.5.0**: Data manipulation and analysis
- **yfinance >= 0.2.0**: Yahoo Finance data access
- **matplotlib >= 3.5.0**: Chart generation
- **ta >= 0.10.0**: Technical analysis indicators
- **xgboost >= 1.6.0**: Machine learning model
- **scikit-learn >= 1.1.0**: ML utilities
- **numpy >= 1.21.0**: Numerical computing
- **scipy >= 1.9.0**: Scientific computing

### Frontend (React)
- **React >= 19.1.0**: UI framework
- **recharts >= 3.0.2**: Chart components
- **lucide-react >= 0.525.0**: Icon library
- **Tailwind CSS**: Utility-first CSS framework

## 🎯 Usage Guide

### 1. Strategy Selection
- Choose from 9 different trading strategies
- Each strategy has unique parameters and risk profiles
- ML strategy requires confidence threshold configuration

### 2. Data Configuration
- **Ticker Symbol**: Enter any valid stock symbol (e.g., AAPL, MSFT, TSLA)
- **Testing Period**: Select predefined periods or custom date ranges
- **ML Parameters**: Configure training periods and confidence levels

### 3. Running Analysis
- Click "Start Backtest" to execute the strategy
- View real-time progress and results
- Examine detailed performance metrics
- Analyze trade-by-trade breakdown

### 4. Interpreting Results
- **Performance Metrics**: Compare strategy vs buy-and-hold
- **Risk Metrics**: Understand volatility and drawdown
- **Trade Analysis**: Review individual trade performance
- **Visual Charts**: See price action with trade markers

## 🧪 Testing

### Running Tests
Navigate to the tests directory and run individual test scripts:
```bash
cd tests
python test_api.py
python test_bollinger.py
python test_connection.py
python Test_1.py
```

### Test Coverage
- **API Testing**: Endpoint validation and error handling
- **Strategy Testing**: Individual strategy performance validation
- **Connection Testing**: Data source connectivity
- **General Testing**: Overall functionality verification

## 📊 Data Sources & Validation

### Primary Data Source
- **Yahoo Finance (yfinance)**: Real-time and historical stock data
- **Data Validation**: Automatic validation of ticker symbols and date ranges
- **Error Handling**: Graceful handling of invalid data requests

### Data Quality
- **Real-time Updates**: Live market data during trading hours
- **Historical Accuracy**: Verified historical price data
- **Missing Data Handling**: Robust handling of gaps and holidays
-  **Machine Learning Model Optimization**: https://docs.google.com/spreadsheets/d/114MO0r3wBKPYmc0iBttv3qy1lljyg4AfW3NgKugEKNg/edit?usp=sharing

## 🔍 Troubleshooting

### Common Issues

1. **Backend Connection Errors**
   - Ensure all Python dependencies are installed: `pip install -r requirements.txt`
   - Check if port 5000 is available
   - Verify Python version (3.8+ required)
   - Check app.log file for detailed error messages

2. **Frontend Connection Issues**
   - Ensure Node.js is installed (version 16+)
   - Run `npm install` in the frontend directory
   - Check if port 3000 is available
   - Clear browser cache and hard refresh

3. **Data Download Problems**
   - Verify internet connection
   - Check if ticker symbol is valid
   - Ensure date range is reasonable
   - Check yfinance API status

4. **Performance Issues**
   - Close other applications to free memory
   - Use shorter date ranges for faster testing
   - Check system resources during ML model training

### Error Logging
- **Backend Logs**: Check `app.log` file for detailed error information
- **Frontend Console**: Use browser developer tools to view JavaScript errors
- **Network Tab**: Monitor API requests and responses

### Port Conflicts
If you encounter port conflicts, modify the ports:
- **Backend**: Edit `backend_api.py` and change `app.run(port=5000)`
- **Frontend**: Edit `frontend/package.json` and add `"PORT": 3001` to scripts

## 🚀 Performance Optimization

### Backend Optimizations
- **Caching**: Implemented for frequently accessed data
- **Async Processing**: Non-blocking API responses
- **Memory Management**: Efficient data handling for large datasets

### Frontend Optimizations
- **Dynamic Loading**: Components load as needed
- **Responsive Design**: Optimized for all screen sizes
- **Efficient Rendering**: Minimal re-renders and updates

## 🔒 Security Considerations

- **Input Validation**: All user inputs are validated
- **Error Handling**: Secure error messages without exposing internals
- **CORS Configuration**: Properly configured for development
- **Data Sanitization**: All data is sanitized before processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Run all test scripts to ensure functionality
5. Update documentation as needed
6. Submit a pull request with detailed description

### Development Guidelines
- Follow existing code style and conventions
- Add appropriate error handling
- Include tests for new features
- Update documentation for any API changes

## 📄 License

This project is for educational and research purposes. Please ensure compliance with data usage terms from Yahoo Finance and other data providers.

## 🆘 Support

### Getting Help
1. Check the troubleshooting section above
2. Review the error logs in `app.log`
3. Test with different ticker symbols and date ranges
4. Create an issue in the repository with:
   - Detailed error description
   - Steps to reproduce
   - System information
   - Error logs

### Known Limitations
- Data availability depends on Yahoo Finance API
- ML model performance varies by market conditions
- Historical data may have gaps during market closures
- Real-time data requires active internet connection

---

**Last Updated**: July 2024
**Version**: 1.0.0  
**Compatibility**: Python 3.8+, Node.js 16+, React 19+ 