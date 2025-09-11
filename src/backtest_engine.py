"""
Backtest Engine Module for Moving Average Crossover Strategy

This module simulates trading based on the generated signals and tracks
portfolio performance, trades, and various metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'LONG' or 'SHORT'
    pnl: float
    return_pct: float
    duration_days: int
    entry_signal: str
    exit_signal: str


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 10000.0
    position_size: float = 1.0  # Fraction of capital to invest
    transaction_cost: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.0  # Price slippage as percentage
    max_drawdown_limit: Optional[float] = None  # Stop trading if drawdown exceeds
    

class BacktestEngine:
    """
    Backtesting engine for moving average crossover strategy.
    
    This class simulates trading based on buy/sell signals and tracks
    portfolio performance over time.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize the backtest engine.
        
        Args:
            config (BacktestConfig): Backtesting configuration
        """
        self.config = config
        self.reset_portfolio()
        
    def reset_portfolio(self):
        """Reset portfolio to initial state."""
        self.initial_capital = self.config.initial_capital
        self.cash = self.initial_capital
        self.position = 0  # Number of shares held
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = []
        self.drawdown_history = []
        self.peak_value = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.is_long = False
        self.entry_price = 0.0
        self.entry_date = None
        
    def calculate_position_size(self, price: float) -> int:
        """
        Calculate the number of shares to buy based on available cash.
        
        Args:
            price (float): Current stock price
            
        Returns:
            int: Number of shares to buy
        """
        available_capital = self.cash * self.config.position_size
        # Account for transaction costs
        effective_price = price * (1 + self.config.transaction_cost + self.config.slippage)
        return int(available_capital / effective_price)
    
    def execute_buy_signal(self, date: datetime, price: float) -> bool:
        """
        Execute a buy signal (open long position).
        
        Args:
            date (datetime): Trade date
            price (float): Execution price
            
        Returns:
            bool: True if trade was executed successfully
        """
        if self.is_long:
            return False  # Already long, skip signal
            
        # Calculate position size
        shares_to_buy = self.calculate_position_size(price)
        
        if shares_to_buy <= 0:
            return False  # Not enough capital
        
        # Calculate total cost including transaction costs
        gross_cost = shares_to_buy * price
        transaction_cost = gross_cost * self.config.transaction_cost
        slippage_cost = gross_cost * self.config.slippage
        total_cost = gross_cost + transaction_cost + slippage_cost
        
        if total_cost > self.cash:
            return False  # Not enough cash
        
        # Execute the trade
        self.cash -= total_cost
        self.position = shares_to_buy
        self.is_long = True
        self.entry_price = price
        self.entry_date = date
        
        return True
    
    def execute_sell_signal(self, date: datetime, price: float) -> bool:
        """
        Execute a sell signal (close long position).
        
        Args:
            date (datetime): Trade date
            price (float): Execution price
            
        Returns:
            bool: True if trade was executed successfully
        """
        if not self.is_long or self.position <= 0:
            return False  # No position to sell
        
        # Calculate proceeds from sale
        gross_proceeds = self.position * price
        transaction_cost = gross_proceeds * self.config.transaction_cost
        slippage_cost = gross_proceeds * self.config.slippage
        net_proceeds = gross_proceeds - transaction_cost - slippage_cost
        
        # Calculate trade P&L
        entry_cost = self.position * self.entry_price * (1 + self.config.transaction_cost)
        pnl = net_proceeds - entry_cost
        return_pct = pnl / entry_cost
        
        # Record the completed trade
        trade = Trade(
            entry_date=self.entry_date,
            exit_date=date,
            entry_price=self.entry_price,
            exit_price=price,
            quantity=self.position,
            side='LONG',
            pnl=pnl,
            return_pct=return_pct,
            duration_days=(date - self.entry_date).days,
            entry_signal='BUY',
            exit_signal='SELL'
        )
        self.trades.append(trade)
        
        # Update portfolio
        self.cash += net_proceeds
        self.position = 0
        self.is_long = False
        self.entry_price = 0.0
        self.entry_date = None
        
        return True
    
    def update_portfolio_value(self, date: datetime, price: float):
        """
        Update portfolio value and track drawdown.
        
        Args:
            date (datetime): Current date
            price (float): Current stock price
        """
        # Calculate current portfolio value
        position_value = self.position * price if self.position > 0 else 0
        self.portfolio_value = self.cash + position_value
        
        # Update peak value and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            
        # Update maximum drawdown
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown
        
        # Record portfolio history
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'position_shares': self.position,
            'drawdown': self.current_drawdown,
            'is_long': self.is_long
        })
        
        # Check drawdown limit
        if (self.config.max_drawdown_limit and 
            self.current_drawdown > self.config.max_drawdown_limit):
            print(f"WARNING: Drawdown limit exceeded on {date}: {self.current_drawdown:.2%}")
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run the complete backtest on the provided data.
        
        Args:
            data (pd.DataFrame): Stock data with signals
            
        Returns:
            Dict: Backtest results and statistics
        """
        print("Starting backtest simulation...")
        self.reset_portfolio()
        
        # Ensure data is sorted by date
        data = data.sort_index()
        
        for date, row in data.iterrows():
            current_price = row['Close']
            
            # Check for buy signal
            if row.get('Buy_Signal', 0) == 1:
                success = self.execute_buy_signal(date, current_price)
                if success:
                    print(f"BUY executed on {date.strftime('%Y-%m-%d')} at ${current_price:.2f}")
            
            # Check for sell signal
            elif row.get('Sell_Signal', 0) == 1:
                success = self.execute_sell_signal(date, current_price)
                if success:
                    print(f"SELL executed on {date.strftime('%Y-%m-%d')} at ${current_price:.2f}")
            
            # Update portfolio value daily
            self.update_portfolio_value(date, current_price)
        
        # Close any open position at the end
        if self.is_long and len(data) > 0:
            final_date = data.index[-1]
            final_price = data.iloc[-1]['Close']
            self.execute_sell_signal(final_date, final_price)
            print(f"Position closed at end of backtest: {final_date.strftime('%Y-%m-%d')} at ${final_price:.2f}")
        
        print(f"Backtest completed. Executed {len(self.trades)} trades.")
        
        return self.get_backtest_results()
    
    def get_backtest_results(self) -> Dict:
        """
        Generate comprehensive backtest results.
        
        Returns:
            Dict: Complete backtest statistics and data
        """
        if not self.portfolio_history:
            return {"error": "No backtest data available"}
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        portfolio_df['cumulative_return'] = (
            portfolio_df['portfolio_value'] / self.initial_capital - 1
        )
        
        # Calculate trade statistics
        trade_stats = self.calculate_trade_statistics()
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(portfolio_df)
        
        return {
            'portfolio_history': portfolio_df,
            'trades': self.trades,
            'trade_statistics': trade_stats,
            'performance_metrics': performance_metrics,
            'config': self.config
        }
    
    def calculate_trade_statistics(self) -> Dict:
        """Calculate statistics for all completed trades."""
        if not self.trades:
            return {"total_trades": 0}
        
        trade_returns = [trade.return_pct for trade in self.trades]
        winning_trades = [t for t in self.trades if t.return_pct > 0]
        losing_trades = [t for t in self.trades if t.return_pct < 0]
        
        stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'average_return': np.mean(trade_returns) if trade_returns else 0,
            'average_winning_return': np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0,
            'average_losing_return': np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0,
            'best_trade': max(trade_returns) if trade_returns else 0,
            'worst_trade': min(trade_returns) if trade_returns else 0,
            'average_holding_period': np.mean([t.duration_days for t in self.trades]) if self.trades else 0,
            'total_pnl': sum([t.pnl for t in self.trades]),
            'profit_factor': abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades])) if losing_trades else float('inf')
        }
        
        return stats
    
    def calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        if portfolio_df.empty:
            return {}
        
        # Basic metrics
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        days = len(portfolio_df)
        years = days / 252  # Trading days per year
        cagr = (final_value / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        daily_returns = portfolio_df['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (cagr - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown metrics
        max_dd = self.max_drawdown
        avg_dd = portfolio_df['drawdown'].mean()
        
        # Additional metrics
        sortino_ratio = self.calculate_sortino_ratio(daily_returns)
        calmar_ratio = cagr / max_dd if max_dd > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd,
            'average_drawdown': avg_dd,
            'total_days': days,
            'trading_years': years
        }
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation version of Sharpe ratio)."""
        if returns.empty:
            return 0
        
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return (returns.mean() * 252 - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0


def run_buy_and_hold_benchmark(data: pd.DataFrame, initial_capital: float, 
                              transaction_cost: float = 0.001) -> Dict:
    """
    Run a buy-and-hold benchmark for comparison.
    
    Args:
        data (pd.DataFrame): Stock price data
        initial_capital (float): Starting capital
        transaction_cost (float): Transaction cost rate
        
    Returns:
        Dict: Buy-and-hold performance metrics
    """
    if data.empty:
        return {}
    
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    
    # Calculate shares bought (accounting for transaction cost)
    effective_start_price = start_price * (1 + transaction_cost)
    shares = int(initial_capital / effective_start_price)
    
    # Calculate final value (accounting for transaction cost on sale)
    gross_proceeds = shares * end_price
    net_proceeds = gross_proceeds * (1 - transaction_cost)
    
    total_return = (net_proceeds - initial_capital) / initial_capital
    days = len(data)
    years = days / 252
    cagr = (net_proceeds / initial_capital) ** (1/years) - 1 if years > 0 else 0
    
    # Calculate daily returns for risk metrics
    price_returns = data['Close'].pct_change().dropna()
    volatility = price_returns.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.02) / volatility if volatility > 0 else 0
    
    # Calculate drawdown
    cumulative = (1 + price_returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'strategy': 'Buy and Hold',
        'initial_capital': initial_capital,
        'final_value': net_proceeds,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'shares_held': shares,
        'start_price': start_price,
        'end_price': end_price
    }


if __name__ == "__main__":
    # Example usage
    from data_fetcher import fetch_stock_data
    from strategy import MovingAverageCrossoverStrategy
    
    # Fetch data and generate signals
    data = fetch_stock_data("AAPL", "2020-01-01", "2024-01-01", 20, 50)
    strategy = MovingAverageCrossoverStrategy(20, 50)
    data_with_signals = strategy.generate_signals(data)
    
    # Configure and run backtest
    config = BacktestConfig(
        initial_capital=10000,
        position_size=1.0,
        transaction_cost=0.001
    )
    
    engine = BacktestEngine(config)
    results = engine.run_backtest(data_with_signals)
    
    # Print results
    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
    print(f"CAGR: {results['performance_metrics']['cagr']:.2%}")
    print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {results['trade_statistics']['win_rate']:.2%}")
    print(f"Total Trades: {results['trade_statistics']['total_trades']}")
    
    # Compare with buy-and-hold
    bh_results = run_buy_and_hold_benchmark(data, 10000)
    print(f"\n=== BUY & HOLD BENCHMARK ===")
    print(f"Total Return: {bh_results['total_return']:.2%}")
    print(f"CAGR: {bh_results['cagr']:.2%}")
    print(f"Sharpe Ratio: {bh_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {bh_results['max_drawdown']:.2%}")