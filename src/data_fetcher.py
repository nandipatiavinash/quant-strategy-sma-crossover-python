"""
Data Fetcher Module for Moving Average Crossover Backtesting

This module handles fetching historical stock price data and calculating
moving averages required for the trading strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataFetcher:
    """
    Handles data fetching and preprocessing for backtesting.
    """
    
    def __init__(self, symbol: str, start_date: str, end_date: str):
        """
        Initialize DataFetcher with stock symbol and date range.
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical stock price data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Stock price data with OHLCV columns
            
        Raises:
            ValueError: If data fetching fails or returns empty dataset
        """
        try:
            print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
                
            # Clean the data
            self.data = self.data.dropna()
            
            print(f"Successfully fetched {len(self.data)} trading days of data")
            return self.data
            
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.symbol}: {str(e)}")
    
    def calculate_moving_averages(self, 
                                short_period: int, 
                                long_period: int,
                                ma_type: str = "SMA") -> pd.DataFrame:
        """
        Calculate short-term and long-term moving averages.
        
        Args:
            short_period (int): Period for short-term MA
            long_period (int): Period for long-term MA
            ma_type (str): Type of moving average ('SMA' or 'EMA')
            
        Returns:
            pd.DataFrame: Data with added moving average columns
            
        Raises:
            ValueError: If data hasn't been fetched or periods are invalid
        """
        if self.data is None:
            raise ValueError("Data must be fetched before calculating moving averages")
            
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")
            
        data_with_ma = self.data.copy()
        
        if ma_type.upper() == "SMA":
            # Simple Moving Average
            data_with_ma[f'MA_{short_period}'] = data_with_ma['Close'].rolling(
                window=short_period
            ).mean()
            data_with_ma[f'MA_{long_period}'] = data_with_ma['Close'].rolling(
                window=long_period
            ).mean()
            
        elif ma_type.upper() == "EMA":
            # Exponential Moving Average
            data_with_ma[f'MA_{short_period}'] = data_with_ma['Close'].ewm(
                span=short_period
            ).mean()
            data_with_ma[f'MA_{long_period}'] = data_with_ma['Close'].ewm(
                span=long_period
            ).mean()
        else:
            raise ValueError("ma_type must be 'SMA' or 'EMA'")
            
        # Remove rows with NaN values (from initial MA calculation period)
        data_with_ma = data_with_ma.dropna()
        
        print(f"Calculated {ma_type} moving averages: {short_period} and {long_period} periods")
        print(f"Data shape after MA calculation: {data_with_ma.shape}")
        
        return data_with_ma
    
    def get_price_data(self) -> Optional[pd.DataFrame]:
        """
        Get the fetched price data.
        
        Returns:
            pd.DataFrame: The fetched price data or None if not fetched
        """
        return self.data
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the integrity of the fetched data.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if all required columns exist
        if not all(col in data.columns for col in required_columns):
            print("Warning: Missing required columns in data")
            return False
            
        # Check for negative prices
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            print("Warning: Found negative or zero prices in data")
            return False
            
        # Check for logical price relationships
        if ((data['High'] < data['Low']) | 
            (data['High'] < data['Open']) | 
            (data['High'] < data['Close']) |
            (data['Low'] > data['Open']) | 
            (data['Low'] > data['Close'])).any():
            print("Warning: Found illogical price relationships in data")
            return False
            
        return True


def fetch_stock_data(symbol: str, 
                    start_date: str, 
                    end_date: str,
                    short_ma: int = 20,
                    long_ma: int = 50,
                    ma_type: str = "SMA") -> pd.DataFrame:
    """
    Convenience function to fetch stock data with moving averages.
    
    Args:
        symbol (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        short_ma (int): Short-term moving average period
        long_ma (int): Long-term moving average period
        ma_type (str): Moving average type ('SMA' or 'EMA')
        
    Returns:
        pd.DataFrame: Stock data with moving averages
    """
    fetcher = DataFetcher(symbol, start_date, end_date)
    
    # Fetch the raw data
    raw_data = fetcher.fetch_data()
    
    # Validate data quality
    if not fetcher.validate_data(raw_data):
        print("Data validation warnings detected. Proceeding with caution...")
    
    # Calculate moving averages
    data_with_ma = fetcher.calculate_moving_averages(short_ma, long_ma, ma_type)
    
    return data_with_ma


if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    data = fetch_stock_data(symbol, start_date, end_date, 20, 50, "SMA")
    print(f"\nData shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nLast few rows:")
    print(data.tail())