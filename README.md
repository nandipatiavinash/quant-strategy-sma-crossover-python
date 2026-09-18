# Quant Strategy Backtester — Moving Average Crossover

A Python-based, event-driven backtesting engine for evaluating moving-average trading strategies against historical market data.

The project focuses on **quantitative analysis, realistic trade simulation, portfolio performance, and risk measurement** rather than simply generating buy/sell signals.

> **Scope:** This is a research and educational backtesting project. Historical backtest results do not guarantee future performance and should not be interpreted as investment advice.

## What it does

The engine lets you configure a trading strategy, simulate historical execution, and evaluate the resulting portfolio using return and risk metrics.

```text
Historical Market Data
        │
        ▼
   Data Handler
        │
        ▼
 Moving Average Strategy
     SMA / EMA
        │
        ▼
    Trade Signals
        │
        ▼
 Backtesting Engine
  ├─ transaction costs
  ├─ slippage
  └─ portfolio state
        │
        ▼
 Performance & Risk Analysis
        │
        ├─ CAGR
        ├─ Sharpe Ratio
        ├─ Sortino Ratio
        ├─ Maximum Drawdown
        ├─ Win Rate
        ├─ Profit Factor
        └─ Calmar Ratio
        │
        ▼
 Visual Reports / Analysis Notebook
```

## Key capabilities

### Historical data

- Downloads historical market data with `yfinance`
- Supports configurable ticker and date ranges
- Prepares time-series data for strategy evaluation

### Strategy engine

Implements:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Configurable short and long windows
- Golden-cross buy signals
- Death-cross sell signals

### Realistic execution model

The simulator accounts for factors that can materially affect a backtest:

- Transaction costs
- Price slippage
- Position state
- Cash balance
- Portfolio value
- Equity curve

### Risk and performance analysis

The engine calculates:

| Metric | Purpose |
|---|---|
| CAGR | Annualized compounded growth |
| Sharpe Ratio | Return relative to volatility |
| Sortino Ratio | Return relative to downside volatility |
| Maximum Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit relative to gross loss |
| Calmar Ratio | Return relative to maximum drawdown |

### Visual analysis

Generated analysis can include:

- Price with moving averages and trade signals
- Portfolio equity curve
- Drawdown over time
- Monthly return heatmap
- Trade outcome analysis

A Jupyter notebook is also provided for interactive exploration.

## Repository structure

```text
quant-strategy-sma-crossover-python/
├── backtester/
│   ├── engine.py           # Core event-driven backtesting logic
│   ├── data_handler.py     # Market data retrieval and preparation
│   ├── strategy.py         # Moving-average strategy
│   ├── portfolio.py        # Positions, cash and portfolio value
│   └── performance.py      # Performance and risk metrics
├── configs/
│   └── config.yaml         # Backtest configuration
├── notebooks/
│   └── analysis.ipynb      # Interactive analysis
├── outputs/                # Generated reports and visualizations
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

## Configuration

Backtest parameters are defined in `configs/config.yaml`, allowing experiments without changing application code.

Example:

```yaml
ticker: 'AAPL'
start_date: '2018-01-01'
end_date: '2023-12-31'
strategy_name: 'MovingAverageCrossover'
short_ma: 50
long_ma: 200
ma_type: 'SMA'
initial_capital: 100000.0
transaction_cost_pct: 0.001
slippage_pct: 0.0005
generate_plots: true
```

## Running locally

### Prerequisites

- Python 3.8+
- pip
- venv

### Setup

```bash
git clone https://github.com/nandipatiavinash/quant-strategy-sma-crossover-python.git
cd quant-strategy-sma-crossover-python

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

Generated reports and charts are written to the `outputs/` directory when enabled.

## Dependencies

- pandas
- numpy
- yfinance
- matplotlib
- seaborn
- scipy
- openpyxl
- pyyaml

## Research considerations

A backtest can be misleading if the methodology ignores execution costs, slippage, data quality, or the difference between in-sample and out-of-sample performance.

This project therefore treats transaction costs and slippage as configurable inputs and exposes multiple risk metrics rather than relying on return alone.

Potential extensions include:

- RSI, MACD and Bollinger Bands
- Parameter optimization
- Walk-forward testing
- Additional order types
- Portfolio-level backtesting across multiple assets
- Expanded unit-test coverage

## Portfolio relevance

This project demonstrates **Python engineering, time-series analysis, quantitative reasoning, financial data processing, portfolio simulation, and risk measurement** through a reproducible backtesting workflow.

## License

MIT License.
