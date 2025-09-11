"""
Strategy Module for Moving Average Crossover Backtesting

This module implements the moving average crossover trading strategy
and generates buy/sell signals based on MA crossovers.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from enum import Enum


class SignalType(Enum):
    """Enumeration for trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0


class MovingAverageCrossoverStrategy:
    """
    Implements the Moving Average Crossover trading strategy.
    
    Strategy Rules:
    - BUY when short-term MA crosses above long-term MA (Golden Cross)
    - SELL when short-term MA crosses below long-term MA (Death Cross)
    - HOLD when no crossover occurs
    """
    
    def __init__(self, short_period: int, long_period: int):
        """
        Initialize the strategy with MA periods.
        
        Args:
            short_period (int): Short-term moving average period
            long_period (int): Long-term moving average period
        """
        self.short_period = short_period
        self.long_period = long_period
        self.signals = None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals based on moving average crossovers.
        
        Args:
            data (pd.DataFrame): Stock data with moving averages
            
        Returns:
            pd.DataFrame: Data with added signal columns
            
        Raises:
            ValueError: If required MA columns are missing
        """
        # Check if required columns exist
        short_ma_col = f'MA_{self.short_period}'
        long_ma_col = f'MA_{self.long_period}'
        
        if short_ma_col not in data.columns or long_ma_col not in data.columns:
            raise ValueError(f"Missing required MA columns: {short_ma_col}, {long_ma_col}")
        
        # Create a copy of the data to avoid modifying original
        data_with_signals = data.copy()
        
        # Calculate crossover signals
        data_with_signals['Short_MA'] = data_with_signals[short_ma_col]
        data_with_signals['Long_MA'] = data_with_signals[long_ma_col]
        
        # Identify crossovers
        # Golden Cross: Short MA crosses above Long MA
        # Death Cross: Short MA crosses below Long MA
        data_with_signals['MA_Diff'] = (
            data_with_signals['Short_MA'] - data_with_signals['Long_MA']
        )
        data_with_signals['MA_Diff_Prev'] = data_with_signals['MA_Diff'].shift(1)
        
        # Generate signals
        conditions = [
            # Buy signal: Short MA crosses above Long MA
            (data_with_signals['MA_Diff'] > 0) & (data_with_signals['MA_Diff_Prev'] <= 0),
            # Sell signal: Short MA crosses below Long MA
            (data_with_signals['MA_Diff'] < 0) & (data_with_signals['MA_Diff_Prev'] >= 0)
        ]
        
        choices = [SignalType.BUY.value, SignalType.SELL.value]
        
        data_with_signals['Signal'] = np.select(conditions, choices, default=SignalType.HOLD.value)
        
        # Create binary signal columns for easier analysis
        data_with_signals['Buy_Signal'] = (data_with_signals['Signal'] == SignalType.BUY.value).astype(int)
        data_with_signals['Sell_Signal'] = (data_with_signals['Signal'] == SignalType.SELL.value).astype(int)
        
        # Store signals for later analysis
        self.signals = data_with_signals
        
        # Print signal summary
        buy_count = data_with_signals['Buy_Signal'].sum()
        sell_count = data_with_signals['Sell_Signal'].sum()
        print(f"Generated {buy_count} BUY signals and {sell_count} SELL signals")
        
        return data_with_signals
    
    def get_signal_details(self) -> Dict[str, any]:
        """
        Get detailed information about generated signals.
        
        Returns:
            Dict: Signal statistics and details
        """
        if self.signals is None:
            return {"error": "Signals not generated yet"}
        
        buy_signals = self.signals[self.signals['Buy_Signal'] == 1]
        sell_signals = self.signals[self.signals['Sell_Signal'] == 1]
        
        return {
            "total_buy_signals": len(buy_signals),
            "total_sell_signals": len(sell_signals),
            "buy_signal_dates": buy_signals.index.tolist(),
            "sell_signal_dates": sell_signals.index.tolist(),
            "buy_signal_prices": buy_signals['Close'].tolist(),
            "sell_signal_prices": sell_signals['Close'].tolist(),
        }
    
    def validate_signals(self, data: pd.DataFrame) -> bool:
        """
        Validate the generated signals for consistency.
        
        Args:
            data (pd.DataFrame): Data with signals
            
        Returns:
            bool: True if signals are valid
        """
        # Check that we don't have simultaneous buy and sell signals
        simultaneous_signals = (
            (data['Buy_Signal'] == 1) & (data['Sell_Signal'] == 1)
        ).sum()
        
        if simultaneous_signals > 0:
            print(f"Warning: Found {simultaneous_signals} simultaneous buy/sell signals")
            return False
        
        # Check signal logic consistency
        short_ma = data['Short_MA']
        long_ma = data['Long_MA']
        buy_signals = data['Buy_Signal'] == 1
        sell_signals = data['Sell_Signal'] == 1
        
        # Buy signals should occur when short MA is above long MA
        invalid_buys = buy_signals & (short_ma <= long_ma)
        if invalid_buys.sum() > 0:
            print(f"Warning: Found {invalid_buys.sum()} invalid buy signals")
            return False
        
        # Sell signals should occur when short MA is below long MA  
        invalid_sells = sell_signals & (short_ma >= long_ma)
        if invalid_sells.sum() > 0:
            print(f"Warning: Found {invalid_sells.sum()} invalid sell signals")
            return False
        
        return True


class Position:
    """Represents a trading position."""
    
    def __init__(self, entry_date, entry_price, quantity, position_type):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.position_type = position_type  # 'LONG' or 'SHORT'
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.return_pct = 0.0
        
    def close_position(self, exit_date, exit_price):
        """Close the position and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.position_type == 'LONG':
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.return_pct = (exit_price - self.entry_price) / self.entry_price
        else:  # SHORT position
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.return_pct = (self.entry_price - exit_price) / self.entry_price
    
    def is_open(self):
        """Check if position is still open."""
        return self.exit_date is None
    
    def to_dict(self):
        """Convert position to dictionary for analysis."""
        return {
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'position_type': self.position_type,
            'pnl': self.pnl,
            'return_pct': self.return_pct,
            'duration_days': (self.exit_date - self.entry_date).days if self.exit_date else None
        }


def analyze_signal_timing(data: pd.DataFrame, short_period: int, long_period: int) -> pd.DataFrame:
    """
    Analyze the timing and quality of generated signals.
    
    Args:
        data (pd.DataFrame): Stock data with signals
        short_period (int): Short MA period
        long_period (int): Long MA period
        
    Returns:
        pd.DataFrame: Analysis results
    """
    buy_signals = data[data['Buy_Signal'] == 1].copy()
    sell_signals = data[data['Sell_Signal'] == 1].copy()
    
    analysis = []
    
    for idx, signal_row in buy_signals.iterrows():
        # Find the next sell signal after this buy signal
        future_sells = sell_signals[sell_signals.index > idx]
        
        if not future_sells.empty:
            next_sell = future_sells.iloc[0]
            holding_period = (next_sell.name - idx).days
            price_change = (next_sell['Close'] - signal_row['Close']) / signal_row['Close']
            
            analysis.append({
                'signal_type': 'BUY',
                'signal_date': idx,
                'signal_price': signal_row['Close'],
                'next_opposite_signal_date': next_sell.name,
                'next_opposite_signal_price': next_sell['Close'],
                'holding_period_days': holding_period,
                'price_change_pct': price_change,
                'short_ma_value': signal_row['Short_MA'],
                'long_ma_value': signal_row['Long_MA'],
                'ma_spread': signal_row['Short_MA'] - signal_row['Long_MA']
            })
    
    return pd.DataFrame(analysis)


if __name__ == "__main__":
    # Example usage with sample data
    from data_fetcher import fetch_stock_data
    
    # Fetch sample data
    data = fetch_stock_data("AAPL", "2020-01-01", "2024-01-01", 20, 50)
    
    # Initialize strategy
    strategy = MovingAverageCrossoverStrategy(20, 50)
    
    # Generate signals
    data_with_signals = strategy.generate_signals(data)
    
    # Validate signals
    is_valid = strategy.validate_signals(data_with_signals)
    print(f"Signals validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Get signal details
    signal_details = strategy.get_signal_details()
    print(f"Signal details: {signal_details}")
    
    # Analyze signal timing
    timing_analysis = analyze_signal_timing(data_with_signals, 20, 50)
    print(f"\nTiming Analysis:")
    print(timing_analysis.head())