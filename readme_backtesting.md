# Python Backtesting Engine for Moving Average Crossover Strategy

A robust, event-driven backtesting system built in Python to evaluate the performance of Moving Average (MA) Crossover trading strategies (SMA/EMA). This tool allows you to simulate trades on historical stock data, incorporating realistic factors like transaction costs and slippage, to gauge a strategy's viability before live deployment.

## 🎯 Purpose

The primary goal of this project is to provide a reliable framework for quantitative analysis of the popular Moving Average Crossover strategy. By backtesting against historical data, users can:

- **Evaluate Profitability**: Determine if the strategy generates positive returns over a specified period
- **Assess Risk**: Analyze key risk metrics like Maximum Drawdown, Sharpe Ratio, and Sortino Ratio
- **Optimize Parameters**: Test different MA periods (e.g., 50 vs. 200, 20 vs. 50) to find optimal settings for a given asset
- **Build Confidence**: Gain data-driven insights before committing real capital

## 📊 Key Features

### Data Fetching
- Seamlessly downloads historical price data using the `yfinance` library

### Flexible Strategy Implementation
- Implements both Simple Moving Average (SMA) and Exponential Moving Average (EMA)
- Generates "Golden Cross" (buy) signals when the short-term MA crosses above the long-term MA
- Generates "Death Cross" (sell) signals when the short-term MA crosses below the long-term MA

### Realistic Backtesting Engine
- Simulates trade execution based on signals
- Tracks portfolio value and equity curve over time
- Accounts for transaction costs and slippage to provide a more accurate performance picture

### In-Depth Performance Analysis
Calculates a comprehensive set of performance metrics, including:
- Compounded Annual Growth Rate (CAGR)
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown
- Win Rate & Profit Factor
- Calmar Ratio

### Rich Visualizations
Generates a suite of plots saved to the `outputs/` directory for easy analysis:
- **Price Chart with MA & Signals**: Visualizes entry and exit points on the price chart
- **Equity Curve**: Tracks portfolio value growth over the backtest period
- **Drawdown Plot**: Highlights periods of portfolio value decline
- **Monthly Returns Heatmap**: Shows strategy performance on a month-by-month basis
- **Trade Analysis**: Provides insights into the distribution of profitable vs. losing trades

## 📂 Repository Structure

The project is organized into logical modules for clarity and scalability.

```
quant-strategy-sma-crossover-python/
│
├── backtester/
│   ├── __init__.py
│   ├── engine.py           # Core backtesting logic
│   ├── data_handler.py     # Fetches and prepares data
│   ├── strategy.py         # Defines the MA Crossover strategy
│   ├── portfolio.py        # Manages positions, cash, and portfolio value
│   └── performance.py      # Calculates and displays performance metrics
│
├── configs/
│   └── config.yaml         # Configuration file for backtest parameters
│
├── outputs/                # Directory for generated charts and reports
│   └── (plots will be saved here)
│
├── main.py                 # Main script to run the backtest
└── requirements.txt        # Python package dependencies
```

## ⚙️ Installation & Quickstart

Get the backtesting engine up and running in a few simple steps.

### Prerequisites
- Python 3.8 or higher
- pip and venv

### 1. Clone the Repository
```bash
git clone https://github.com/nandipatiavinash/quant-strategy-sma-crossover-python.git
cd quant-strategy-sma-crossover-python
```

### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Required Packages
Install all the necessary libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run the Backtest
Execute the `main.py` script to run a backtest using the default parameters defined in `configs/config.yaml`.

```bash
python main.py
```

After the run completes, check the `outputs/` folder for the performance report and visualizations.

### 5. Run with Custom Parameters
You can easily modify the backtest parameters by editing the `configs/config.yaml` file. No code changes are needed!

## 🛠️ Configuration (configs/config.yaml)

All backtest parameters are controlled via a simple YAML file. This allows for easy experimentation without touching the source code.

```yaml
# configs/config.yaml

# Data Parameters
ticker: 'AAPL'          # Stock ticker symbol (e.g., 'GOOGL', 'MSFT', 'BTC-USD')
start_date: '2018-01-01'
end_date: '2023-12-31'

# Strategy Parameters
strategy_name: 'MovingAverageCrossover'
short_ma: 50              # Short-term moving average window
long_ma: 200              # Long-term moving average window
ma_type: 'SMA'            # Type of MA: 'SMA' or 'EMA'

# Portfolio & Risk Parameters
initial_capital: 100000.0 # Starting portfolio value in USD
transaction_cost_pct: 0.001 # 0.1% cost per trade (e.g., broker fees)
slippage_pct: 0.0005        # 0.05% price slippage per trade

# Output Settings
generate_plots: true      # Set to false to disable chart generation
```

## 📦 Dependencies (requirements.txt)

This file lists all the Python libraries required by the project.

```txt
pandas
numpy
yfinance
matplotlib
seaborn
scipy
openpyxl
pyyaml
```

## ❓ Common Issues & Fixes

### Issue: `ModuleNotFoundError: No module named 'yaml'`
- **Reason**: The `pyyaml` library is not installed correctly
- **Solution**: Make sure you have activated your virtual environment and run `pip install pyyaml` or `pip install -r requirements.txt`

### Issue: yfinance fails to download data (e.g., `[...]: No data found, symbol may be delisted`)
- **Reason**: The ticker symbol might be incorrect, delisted, or not available for the specified date range
- **Solution**: Double-check the ticker in `config.yaml`. Try a different symbol (e.g., 'SPY') or adjust the `start_date` and `end_date`

## 📝 Future Enhancements

This project provides a solid foundation. Here are some ideas for future improvements:

- **Add More Indicators**: Integrate other technical indicators like RSI, MACD, or Bollinger Bands to create more complex strategies
- **Parameter Optimization**: Implement a grid search or randomized search to automatically find the best-performing MA windows
- **Improved Execution Model**: Simulate order types like limit and stop-loss orders for more realistic backtesting
- **Unit Testing**: Add a suite of unit tests to ensure the reliability and accuracy of the backtesting engine's components
- **Portfolio-Level Backtesting**: Extend the engine to test strategies across a portfolio of multiple assets simultaneously

## 👨‍💻 Author & Contact

This project was created by **Avi Nandipati**.

- **Email**: nandipatiavinash19@gmail.com
- **GitHub**: [@nandipatiavinash](https://github.com/nandipatiavinash)

Feel free to reach out with any questions, suggestions, or collaboration opportunities!

## 📜 License

This project is licensed under the MIT License.