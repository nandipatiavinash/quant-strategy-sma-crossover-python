"""
Visualization Module for Moving Average Crossover Backtesting

This module provides comprehensive visualization capabilities for
backtesting results, including price charts, equity curves, and performance analytics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BacktestVisualizer:
    """
    Comprehensive visualization suite for backtesting results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'price': '#1f77b4',
            'short_ma': '#ff7f0e',
            'long_ma': '#2ca02c',
            'buy_signal': '#d62728',
            'sell_signal': '#9467bd',
            'portfolio': '#8c564b',
            'benchmark': '#e377c2',
            'drawdown': '#7f7f7f'
        }
    
    def plot_price_and_signals(self, data: pd.DataFrame, 
                             short_period: int, long_period: int,
                             title: str = "Price Chart with Moving Averages and Signals",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot stock price with moving averages and buy/sell signals.
        
        Args:
            data (pd.DataFrame): Stock data with signals
            short_period (int): Short MA period
            long_period (int): Long MA period
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot price and moving averages
        ax.plot(data.index, data['Close'], label='Price', 
                color=self.colors['price'], linewidth=1.5, alpha=0.8)
        ax.plot(data.index, data['Short_MA'], label=f'MA {short_period}', 
                color=self.colors['short_ma'], linewidth=2)
        ax.plot(data.index, data['Long_MA'], label=f'MA {long_period}', 
                color=self.colors['long_ma'], linewidth=2)
        
        # Plot buy signals
        buy_signals = data[data['Buy_Signal'] == 1]
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['Close'], 
                      marker='^', color=self.colors['buy_signal'], 
                      s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals
        sell_signals = data[data['Sell_Signal'] == 1]
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['Close'], 
                      marker='v', color=self.colors['sell_signal'], 
                      s=100, label='Sell Signal', zorder=5)
        
        # Formatting
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Price chart saved to: {save_path}")
        
        return fig
    
    def plot_equity_curve(self, portfolio_data: pd.DataFrame, 
                         benchmark_data: Optional[pd.DataFrame] = None,
                         title: str = "Portfolio Equity Curve",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot portfolio equity curve with optional benchmark comparison.
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio performance data
            benchmark_data (pd.DataFrame, optional): Benchmark data for comparison
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                      height_ratios=[3, 1], sharex=True)
        
        # Plot equity curve
        ax1.plot(portfolio_data.index, portfolio_data['portfolio_value'], 
                label='Strategy Portfolio', color=self.colors['portfolio'], linewidth=2)
        
        # Add benchmark if provided
        if benchmark_data is not None:
            initial_value = portfolio_data['portfolio_value'].iloc[0]
            benchmark_normalized = benchmark_data['Close'] / benchmark_data['Close'].iloc[0] * initial_value
            ax1.plot(benchmark_data.index, benchmark_normalized, 
                    label='Buy & Hold Benchmark', color=self.colors['benchmark'], 
                    linewidth=2, linestyle='--', alpha=0.8)
        
        # Formatting for equity curve
        ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        if 'drawdown' in portfolio_data.columns:
            drawdown = portfolio_data['drawdown'] * 100  # Convert to percentage
            ax2.fill_between(portfolio_data.index, drawdown, 0, 
                           color=self.colors['drawdown'], alpha=0.3, label='Drawdown')
            ax2.plot(portfolio_data.index, drawdown, 
                    color=self.colors['drawdown'], linewidth=1)
        
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Drawdown should go downward
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to: {save_path}")
        
        return fig
    
    def plot_performance_metrics(self, performance_report: Dict,
                               title: str = "Performance Metrics Summary",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive performance metrics visualization.
        
        Args:
            performance_report (Dict): Performance analysis report
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create a 2x3 subplot layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
        
        # 1. Return Metrics Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        returns = performance_report.get('returns', {})
        return_metrics = {
            'Total Return': returns.get('total_return', 0) * 100,
            'CAGR': returns.get('cagr', 0) * 100,
            'Volatility': returns.get('annual_volatility', 0) * 100
        }
        
        bars1 = ax1.bar(return_metrics.keys(), return_metrics.values(), 
                       color=['green' if v > 0 else 'red' for v in return_metrics.values()])
        ax1.set_title('Return Metrics (%)', fontweight='bold')
        ax1.set_ylabel('Percentage')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, return_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 2. Risk-Adjusted Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        risk_metrics = {
            'Sharpe Ratio': returns.get('sharpe_ratio', 0),
            'Sortino Ratio': returns.get('sortino_ratio', 0),
            'Calmar Ratio': performance_report.get('additional_ratios', {}).get('calmar_ratio', 0)
        }
        
        bars2 = ax2.bar(risk_metrics.keys(), risk_metrics.values(), 
                       color=['blue' if v > 0 else 'orange' for v in risk_metrics.values()])
        ax2.set_title('Risk-Adjusted Ratios', fontweight='bold')
        ax2.set_ylabel('Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, risk_metrics.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Trade Analysis (if available)
        ax3 = fig.add_subplot(gs[1, 0])
        trades = performance_report.get('trades', {})
        if trades and trades.get('total_trades', 0) > 0:
            trade_data = {
                'Win Rate': trades.get('win_rate', 0) * 100,
                'Profit Factor': trades.get('profit_factor', 0),
                'Avg Return': trades.get('average_trade_return', 0) * 100
            }
            
            bars3 = ax3.bar(trade_data.keys(), trade_data.values(), 
                           color=['green', 'blue', 'purple'])
            ax3.set_title('Trade Statistics', fontweight='bold')
            ax3.set_ylabel('Value')
            
            for bar, value in zip(bars3, trade_data.values()):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No Trade Data Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Trade Statistics', fontweight='bold')
        
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        ax4 = fig.add_subplot(gs[1, 1])
        drawdown = performance_report.get('drawdown', {})
        dd_data = {
            'Max DD': drawdown.get('max_drawdown', 0) * 100,
            'Avg DD': drawdown.get('average_drawdown', 0) * 100,
            'DD Periods': drawdown.get('drawdown_periods', 0)
        }
        
        bars4 = ax4.bar(dd_data.keys(), dd_data.values(), color=['red', 'orange', 'yellow'])
        ax4.set_title('Drawdown Analysis', fontweight='bold')
        ax4.set_ylabel('Percentage / Count')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, dd_data.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 5. Risk Distribution (spans bottom row)
        ax5 = fig.add_subplot(gs[2, :])
        risk = performance_report.get('risk', {})
        
        # Create risk distribution data
        risk_labels = ['VaR 95%', 'CVaR 95%', 'Max Daily Loss', 'Max Daily Gain']
        risk_values = [
            risk.get('var_95', 0) * 100,
            risk.get('cvar_95', 0) * 100,
            risk.get('max_daily_loss', 0) * 100,
            risk.get('max_daily_gain', 0) * 100
        ]
        
        colors = ['red' if v < 0 else 'green' for v in risk_values]
        bars5 = ax5.bar(risk_labels, risk_values, color=colors, alpha=0.7)
        ax5.set_title('Risk Distribution (%)', fontweight='bold')
        ax5.set_ylabel('Daily Return (%)')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        for bar, value in zip(bars5, risk_values):
            ax5.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.1 if value > 0 else -0.3), 
                    f'{value:.2f}%', ha='center', va='bottom' if value > 0 else 'top')
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance metrics chart saved to: {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades: List, 
                          title: str = "Trade Analysis",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed trade analysis visualizations.
        
        Args:
            trades (List): List of completed trades
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if not trades:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Trades Available for Analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontweight='bold')
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Extract trade data
        trade_returns = [trade.return_pct * 100 for trade in trades]
        trade_durations = [trade.duration_days for trade in trades]
        trade_pnls = [trade.pnl for trade in trades]
        
        # 1. Trade Returns Distribution
        ax1.hist(trade_returns, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(trade_returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(trade_returns):.2f}%')
        ax1.set_title('Trade Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Trade P&L
        cumulative_pnl = np.cumsum(trade_pnls)
        ax2.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, 
                marker='o', linewidth=2, markersize=4)
        ax2.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Trade Duration Analysis
        ax3.hist(trade_durations, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(np.mean(trade_durations), color='red', linestyle='--',
                   label=f'Mean: {np.mean(trade_durations):.1f} days')
        ax3.set_title('Trade Duration Distribution', fontweight='bold')
        ax3.set_xlabel('Duration (Days)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Return vs Duration Scatter
        wins = [i for i, ret in enumerate(trade_returns) if ret > 0]
        losses = [i for i, ret in enumerate(trade_returns) if ret <= 0]
        
        if wins:
            ax4.scatter([trade_durations[i] for i in wins], 
                       [trade_returns[i] for i in wins],
                       color='green', alpha=0.6, label='Winning Trades', s=50)
        if losses:
            ax4.scatter([trade_durations[i] for i in losses], 
                       [trade_returns[i] for i in losses],
                       color='red', alpha=0.6, label='Losing Trades', s=50)
        
        ax4.set_title('Return vs Duration', fontweight='bold')
        ax4.set_xlabel('Duration (Days)')
        ax4.set_ylabel('Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trade analysis chart saved to: {save_path}")
        
        return fig
    
    def plot_monthly_returns_heatmap(self, portfolio_data: pd.DataFrame,
                                   title: str = "Monthly Returns Heatmap",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a monthly returns heatmap.
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio performance data
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate monthly returns
        monthly_returns = portfolio_data['portfolio_value'].resample('M').last().pct_change()
        monthly_returns = monthly_returns.dropna()
        
        # Create pivot table for heatmap
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = monthly_returns.to_frame('returns')
        monthly_data['year'] = monthly_data.index.year
        monthly_data['month'] = monthly_data.index.month
        
        # Pivot to create matrix
        heatmap_data = monthly_data.pivot(index='year', columns='month', values='returns')
        
        # Convert to percentage
        heatmap_data = heatmap_data * 100
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
        
        # Customize month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Monthly returns heatmap saved to: {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, data: pd.DataFrame, 
                                  performance_report: Dict,
                                  trades: List,
                                  portfolio_data: pd.DataFrame,
                                  benchmark_data: Optional[pd.DataFrame] = None,
                                  short_period: int = 20,
                                  long_period: int = 50,
                                  save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive visual report of the backtesting results.
        
        Args:
            data (pd.DataFrame): Stock data with signals
            performance_report (Dict): Performance analysis report
            trades (List): List of completed trades
            portfolio_data (pd.DataFrame): Portfolio performance data
            benchmark_data (pd.DataFrame, optional): Benchmark data
            short_period (int): Short MA period
            long_period (int): Long MA period
            save_dir (str, optional): Directory to save plots
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of created figures
        """
        figures = {}
        
        # 1. Price and Signals Chart
        print("Creating price and signals chart...")
        fig1 = self.plot_price_and_signals(
            data, short_period, long_period,
            title=f"Moving Average Crossover Strategy - {data.index[0].strftime('%Y')} to {data.index[-1].strftime('%Y')}",
            save_path=f"{save_dir}/price_signals.png" if save_dir else None
        )
        figures['price_signals'] = fig1
        
        # 2. Equity Curve
        print("Creating equity curve...")
        fig2 = self.plot_equity_curve(
            portfolio_data, benchmark_data,
            title="Strategy vs Buy & Hold Performance",
            save_path=f"{save_dir}/equity_curve.png" if save_dir else None
        )
        figures['equity_curve'] = fig2
        
        # 3. Performance Metrics
        print("Creating performance metrics chart...")
        fig3 = self.plot_performance_metrics(
            performance_report,
            title="Comprehensive Performance Analysis",
            save_path=f"{save_dir}/performance_metrics.png" if save_dir else None
        )
        figures['performance_metrics'] = fig3
        
        # 4. Trade Analysis
        print("Creating trade analysis...")
        fig4 = self.plot_trade_analysis(
            trades,
            title="Detailed Trade Analysis",
            save_path=f"{save_dir}/trade_analysis.png" if save_dir else None
        )
        figures['trade_analysis'] = fig4
        
        # 5. Monthly Returns Heatmap
        print("Creating monthly returns heatmap...")
        fig5 = self.plot_monthly_returns_heatmap(
            portfolio_data,
            title="Monthly Returns Distribution",
            save_path=f"{save_dir}/monthly_heatmap.png" if save_dir else None
        )
        figures['monthly_heatmap'] = fig5
        
        print(f"Created {len(figures)} visualization charts.")
        return figures
    
    def plot_rolling_metrics(self, portfolio_data: pd.DataFrame,
                           window: int = 252,
                           title: str = "Rolling Performance Metrics",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot rolling performance metrics over time.
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio performance data
            window (int): Rolling window size (default: 252 trading days = 1 year)
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Calculate rolling metrics
        returns = portfolio_data['portfolio_value'].pct_change().dropna()
        
        # Rolling Sharpe Ratio
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2)
        ax1.set_title(f'Rolling Sharpe Ratio ({window//252}-Year)', fontweight='bold')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Rolling Volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol.values, color='orange', linewidth=2)
        ax2.set_title(f'Rolling Volatility ({window//252}-Year)', fontweight='bold')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Maximum Drawdown
        portfolio_values = portfolio_data['portfolio_value']
        rolling_peak = portfolio_values.rolling(window, min_periods=1).max()
        rolling_dd = ((portfolio_values - rolling_peak) / rolling_peak * 100).rolling(window).min()
        ax3.fill_between(rolling_dd.index, rolling_dd.values, 0, 
                        color='red', alpha=0.3)
        ax3.plot(rolling_dd.index, rolling_dd.values, color='red', linewidth=2)
        ax3.set_title(f'Rolling Maximum Drawdown ({window//252}-Year)', fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Rolling Returns
        rolling_returns = portfolio_values.pct_change(window) * 100
        ax4.plot(rolling_returns.index, rolling_returns.values, color='green', linewidth=2)
        ax4.set_title(f'Rolling Returns ({window//252}-Year)', fontweight='bold')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling metrics chart saved to: {save_path}")
        
        return fig
    
    def plot_correlation_analysis(self, portfolio_data: pd.DataFrame,
                                benchmark_data: Optional[pd.DataFrame] = None,
                                title: str = "Correlation Analysis",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation analysis between strategy and benchmark.
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio performance data
            benchmark_data (pd.DataFrame, optional): Benchmark data
            title (str): Plot title
            save_path (str, optional): Path to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if benchmark_data is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No Benchmark Data Available for Correlation Analysis',
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontweight='bold')
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Align data by dates
        portfolio_returns = portfolio_data['portfolio_value'].pct_change().dropna()
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Merge on common dates
        combined_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        combined_data.columns = ['Strategy', 'Benchmark']
        combined_data = combined_data.dropna()
        
        if combined_data.empty:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'No Overlapping Data',
                       ha='center', va='center', transform=ax.transAxes)
            plt.suptitle(title, fontsize=16, fontweight='bold')
            return fig
        
        # 1. Scatter plot of daily returns
        ax1.scatter(combined_data['Benchmark'] * 100, combined_data['Strategy'] * 100, 
                   alpha=0.6, s=20)
        
        # Add regression line
        z = np.polyfit(combined_data['Benchmark'], combined_data['Strategy'], 1)
        p = np.poly1d(z)
        ax1.plot(combined_data['Benchmark'] * 100, p(combined_data['Benchmark']) * 100, 
                "r--", alpha=0.8, linewidth=2)
        
        correlation = combined_data['Strategy'].corr(combined_data['Benchmark'])
        ax1.set_title(f'Daily Returns Correlation: {correlation:.3f}', fontweight='bold')
        ax1.set_xlabel('Benchmark Daily Return (%)')
        ax1.set_ylabel('Strategy Daily Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling correlation
        rolling_corr = combined_data['Strategy'].rolling(window=60).corr(combined_data['Benchmark'])
        ax2.plot(rolling_corr.index, rolling_corr.values, linewidth=2)
        ax2.set_title('60-Day Rolling Correlation', fontweight='bold')
        ax2.set_ylabel('Correlation')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Return distribution comparison
        ax3.hist(combined_data['Strategy'] * 100, bins=30, alpha=0.5, 
                label='Strategy', density=True)
        ax3.hist(combined_data['Benchmark'] * 100, bins=30, alpha=0.5, 
                label='Benchmark', density=True)
        ax3.set_title('Return Distribution Comparison', fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative returns comparison
        strategy_cumulative = (1 + combined_data['Strategy']).cumprod()
        benchmark_cumulative = (1 + combined_data['Benchmark']).cumprod()
        
        ax4.plot(strategy_cumulative.index, strategy_cumulative.values, 
                label='Strategy', linewidth=2)
        ax4.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                label='Benchmark', linewidth=2, linestyle='--')
        ax4.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax4.set_ylabel('Cumulative Return')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format dates
        for ax in [ax2, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation analysis chart saved to: {save_path}")
        
        return fig


def save_all_plots(figures: Dict[str, plt.Figure], directory: str):
    """
    Save all figures to a specified directory.
    
    Args:
        figures (Dict[str, plt.Figure]): Dictionary of figures to save
        directory (str): Directory path to save plots
    """
    import os
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = os.path.join(directory, f"{name}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {name} to {filepath}")
    
    print(f"All plots saved to directory: {directory}")


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


if __name__ == "__main__":
    # Example usage with sample data
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Sample stock data
    prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
    short_ma = pd.Series(prices).rolling(20).mean()
    long_ma = pd.Series(prices).rolling(50).mean()
    
    stock_data = pd.DataFrame({
        'Close': prices,
        'Short_MA': short_ma,
        'Long_MA': long_ma,
        'Buy_Signal': np.random.choice([0, 1], len(dates), p=[0.95, 0.05]),
        'Sell_Signal': np.random.choice([0, 1], len(dates), p=[0.95, 0.05])
    }, index=dates)
    
    # Sample portfolio data
    portfolio_values = 10000 * np.cumprod(1 + np.random.normal(0.0008, 0.015, len(dates)))
    portfolio_data = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'drawdown': np.random.uniform(-0.1, 0, len(dates))
    }, index=dates)
    
    # Sample performance report
    performance_report = {
        'returns': {
            'total_return': 0.25,
            'cagr': 0.08,
            'annual_volatility': 0.15,
            'sharpe_ratio': 0.8,
            'sortino_ratio': 1.2
        },
        'drawdown': {
            'max_drawdown': 0.12,
            'average_drawdown': 0.03,
            'drawdown_periods': 5
        },
        'risk': {
            'var_95': -0.025,
            'cvar_95': -0.035,
            'max_daily_loss': -0.08,
            'max_daily_gain': 0.06
        },
        'trades': {
            'total_trades': 15,
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'average_trade_return': 0.03
        },
        'additional_ratios': {
            'calmar_ratio': 0.67,
            'sterling_ratio': 2.67
        }
    }
    
    # Initialize visualizer
    visualizer = BacktestVisualizer()
    
    # Create sample plots
    print("Creating sample visualizations...")
    
    # Price and signals chart
    fig1 = visualizer.plot_price_and_signals(stock_data, 20, 50)
    
    # Equity curve
    fig2 = visualizer.plot_equity_curve(portfolio_data)
    
    # Performance metrics
    fig3 = visualizer.plot_performance_metrics(performance_report)
    
    # Monthly heatmap
    fig4 = visualizer.plot_monthly_returns_heatmap(portfolio_data)
    
    print("Sample visualizations created successfully!")
    
    # Show plots
    plt.show()