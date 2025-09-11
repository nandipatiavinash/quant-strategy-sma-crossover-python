"""
Performance Metrics Module for Moving Average Crossover Backtesting

This module provides comprehensive performance analysis and risk metrics
for evaluating trading strategy effectiveness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    """
    
    def __init__(self, portfolio_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None):
        """
        Initialize performance analyzer.
        
        Args:
            portfolio_data (pd.DataFrame): Portfolio performance data
            benchmark_data (pd.DataFrame, optional): Benchmark performance data
        """
        self.portfolio_data = portfolio_data.copy()
        self.benchmark_data = benchmark_data.copy() if benchmark_data is not None else None
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def calculate_returns_metrics(self) -> Dict:
        """
        Calculate comprehensive return-based performance metrics.
        
        Returns:
            Dict: Return metrics including total, CAGR, volatility, etc.
        """
        if 'portfolio_value' not in self.portfolio_data.columns:
            raise ValueError("Portfolio data must contain 'portfolio_value' column")
        
        portfolio_values = self.portfolio_data['portfolio_value']
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Basic return metrics
        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time-based metrics
        days = len(portfolio_values)
        years = days / 252  # Trading days per year
        cagr = (final_value / initial_value) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility metrics
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = daily_returns - self.risk_free_rate/252
        sharpe_ratio = excess_returns.mean() / daily_vol * np.sqrt(252) if daily_vol > 0 else 0
        
        # Downside metrics
        negative_returns = daily_returns[daily_returns < 0]
        downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (cagr - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'daily_volatility': daily_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'initial_value': initial_value,
            'final_value': final_value,
            'trading_days': days,
            'trading_years': years
        }
    
    def calculate_drawdown_metrics(self) -> Dict:
        """
        Calculate detailed drawdown analysis.
        
        Returns:
            Dict: Drawdown metrics including max, average, duration, etc.
        """
        portfolio_values = self.portfolio_data['portfolio_value']
        
        # Calculate running maximum (peak)
        peak = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - peak) / peak
        drawdown_pct = drawdown * 100
        
        # Drawdown metrics
        max_drawdown = abs(drawdown.min())
        avg_drawdown = abs(drawdown[drawdown < 0].mean()) if (drawdown < 0).any() else 0
        
        # Drawdown duration analysis
        drawdown_periods = self._identify_drawdown_periods(drawdown)
        
        # Recovery analysis
        underwater_periods = self._calculate_underwater_periods(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': avg_drawdown,
            'drawdown_periods': len(drawdown_periods),
            'longest_drawdown_days': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'average_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
            'underwater_periods': underwater_periods,
            'recovery_factor': abs(self.calculate_returns_metrics()['total_return'] / max_drawdown) if max_drawdown > 0 else float('inf'),
            'drawdown_series': drawdown,
            'drawdown_periods_detail': drawdown_periods
        }
    
    def _identify_drawdown_periods(self, drawdown: pd.Series) -> List[Dict]:
        """Identify distinct drawdown periods."""
        periods = []
        in_drawdown = False
        start_date = None
        start_value = None
        min_value = None
        min_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # Start of drawdown (0.1% threshold)
                in_drawdown = True
                start_date = date
                start_value = dd
                min_value = dd
                min_date = date
            elif dd < -0.001 and in_drawdown:  # Continue drawdown
                if dd < min_value:
                    min_value = dd
                    min_date = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                duration = (date - start_date).days
                periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'max_drawdown': abs(min_value),
                    'max_drawdown_date': min_date
                })
        
        return periods
    
    def _calculate_underwater_periods(self, drawdown: pd.Series) -> Dict:
        """Calculate time spent underwater (below previous peak)."""
        underwater_days = (drawdown < -0.001).sum()  # Days with >0.1% drawdown
        total_days = len(drawdown)
        underwater_percentage = underwater_days / total_days if total_days > 0 else 0
        
        return {
            'underwater_days': underwater_days,
            'total_days': total_days,
            'underwater_percentage': underwater_percentage
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate comprehensive risk metrics.
        
        Returns:
            Dict: Risk metrics including VaR, CVaR, beta, etc.
        """
        daily_returns = self.portfolio_data['portfolio_value'].pct_change().dropna()
        
        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        cvar_99 = daily_returns[daily_returns <= var_99].mean()
        
        # Maximum daily loss and gain
        max_daily_loss = daily_returns.min()
        max_daily_gain = daily_returns.max()
        
        # Skewness and Kurtosis
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        # Tail ratio
        positive_returns = daily_returns[daily_returns > 0]
        negative_returns = daily_returns[daily_returns < 0]
        tail_ratio = (np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))) if len(negative_returns) > 0 else float('inf')
        
        risk_metrics = {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_daily_loss': max_daily_loss,
            'max_daily_gain': max_daily_gain,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'positive_days': len(positive_returns),
            'negative_days': len(negative_returns),
            'flat_days': len(daily_returns[daily_returns == 0])
        }
        
        # Add beta if benchmark data is available
        if self.benchmark_data is not None:
            beta_metrics = self._calculate_beta_metrics(daily_returns)
            risk_metrics.update(beta_metrics)
        
        return risk_metrics
    
    def _calculate_beta_metrics(self, portfolio_returns: pd.Series) -> Dict:
        """Calculate beta and related metrics against benchmark."""
        if self.benchmark_data is None:
            return {}
        
        # Align dates
        benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['portfolio', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 2:
            return {'beta': 0, 'alpha': 0, 'correlation': 0, 'r_squared': 0}
        
        # Calculate beta
        covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0][1]
        benchmark_variance = np.var(aligned_data['benchmark'])
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha (Jensen's alpha)
        portfolio_mean = aligned_data['portfolio'].mean() * 252
        benchmark_mean = aligned_data['benchmark'].mean() * 252
        alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
        # Calculate correlation and R-squared
        correlation = aligned_data['portfolio'].corr(aligned_data['benchmark'])
        r_squared = correlation ** 2
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': (aligned_data['portfolio'] - aligned_data['benchmark']).std() * np.sqrt(252)
        }
    
    def calculate_trade_metrics(self, trades: List) -> Dict:
        """
        Calculate trade-specific performance metrics.
        
        Args:
            trades (List): List of completed trades
            
        Returns:
            Dict: Trade analysis metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_trade_return': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'average_holding_period': 0,
                'sharpe_ratio_trades': 0
            }
        
        # Extract trade data
        trade_returns = [trade.return_pct for trade in trades]
        trade_pnls = [trade.pnl for trade in trades]
        holding_periods = [trade.duration_days for trade in trades]
        
        # Basic trade statistics
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        # Profit factor
        gross_profit = sum([pnl for pnl in trade_pnls if pnl > 0])
        gross_loss = abs(sum([pnl for pnl in trade_pnls if pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Return statistics
        avg_return = np.mean(trade_returns)
        best_trade = max(trade_returns)
        worst_trade = min(trade_returns)
        
        # Risk-adjusted trade metrics
        trade_std = np.std(trade_returns)
        sharpe_trades = avg_return / trade_std if trade_std > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(trade_returns)
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_trade_return': avg_return,
            'average_winning_return': np.mean(winning_trades) if winning_trades else 0,
            'average_losing_return': np.mean(losing_trades) if losing_trades else 0,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'trade_return_std': trade_std,
            'sharpe_ratio_trades': sharpe_trades,
            'average_holding_period': np.mean(holding_periods),
            'median_holding_period': np.median(holding_periods),
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_pnl': sum(trade_pnls)
        }
    
    def _calculate_consecutive_trades(self, trade_returns: List[float]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for return_val in trade_returns:
            if return_val > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif return_val < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:  # Break even trade
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def generate_performance_report(self, trades: List = None) -> Dict:
        """
        Generate a comprehensive performance report.
        
        Args:
            trades (List, optional): List of completed trades
            
        Returns:
            Dict: Complete performance analysis
        """
        report = {}
        
        # Calculate all metrics
        report['returns'] = self.calculate_returns_metrics()
        report['drawdown'] = self.calculate_drawdown_metrics()
        report['risk'] = self.calculate_risk_metrics()
        
        if trades:
            report['trades'] = self.calculate_trade_metrics(trades)
        
        # Calculate additional ratios
        returns_metrics = report['returns']
        drawdown_metrics = report['drawdown']
        
        # Calmar Ratio
        calmar_ratio = returns_metrics['cagr'] / drawdown_metrics['max_drawdown'] if drawdown_metrics['max_drawdown'] > 0 else float('inf')
        
        # Sterling Ratio (alternative risk-adjusted return measure)
        sterling_ratio = returns_metrics['cagr'] / drawdown_metrics['average_drawdown'] if drawdown_metrics['average_drawdown'] > 0 else float('inf')
        
        report['additional_ratios'] = {
            'calmar_ratio': calmar_ratio,
            'sterling_ratio': sterling_ratio,
            'recovery_factor': drawdown_metrics['recovery_factor']
        }
        
        return report
    
    def compare_with_benchmark(self, benchmark_metrics: Dict) -> Dict:
        """
        Compare strategy performance with benchmark.
        
        Args:
            benchmark_metrics (Dict): Benchmark performance metrics
            
        Returns:
            Dict: Comparison analysis
        """
        strategy_metrics = self.calculate_returns_metrics()
        
        comparison = {
            'excess_return': strategy_metrics['total_return'] - benchmark_metrics.get('total_return', 0),
            'excess_cagr': strategy_metrics['cagr'] - benchmark_metrics.get('cagr', 0),
            'return_ratio': strategy_metrics['total_return'] / benchmark_metrics.get('total_return', 1) if benchmark_metrics.get('total_return', 0) != 0 else float('inf'),
            'volatility_ratio': strategy_metrics['annual_volatility'] / benchmark_metrics.get('volatility', 1) if benchmark_metrics.get('volatility', 0) != 0 else float('inf'),
            'sharpe_difference': strategy_metrics['sharpe_ratio'] - benchmark_metrics.get('sharpe_ratio', 0),
            'outperformance_days': None  # Would need daily data to calculate
        }
        
        return comparison


def format_performance_summary(performance_report: Dict) -> str:
    """
    Format performance report into a readable summary.
    
    Args:
        performance_report (Dict): Performance analysis report
        
    Returns:
        str: Formatted performance summary
    """
    returns = performance_report.get('returns', {})
    drawdown = performance_report.get('drawdown', {})
    risk = performance_report.get('risk', {})
    trades = performance_report.get('trades', {})
    ratios = performance_report.get('additional_ratios', {})
    
    summary = f"""
=== STRATEGY PERFORMANCE SUMMARY ===

📈 RETURN METRICS
Total Return:        {returns.get('total_return', 0):.2%}
CAGR:               {returns.get('cagr', 0):.2%}
Annual Volatility:   {returns.get('annual_volatility', 0):.2%}
Sharpe Ratio:       {returns.get('sharpe_ratio', 0):.2f}
Sortino Ratio:      {returns.get('sortino_ratio', 0):.2f}

📉 RISK METRICS
Max Drawdown:       {drawdown.get('max_drawdown', 0):.2%}
Average Drawdown:   {drawdown.get('average_drawdown', 0):.2%}
VaR (95%):          {risk.get('var_95', 0):.2%}
CVaR (95%):         {risk.get('cvar_95', 0):.2%}
Max Daily Loss:     {risk.get('max_daily_loss', 0):.2%}

🎯 TRADE METRICS
Total Trades:       {trades.get('total_trades', 0)}
Win Rate:           {trades.get('win_rate', 0):.2%}
Profit Factor:      {trades.get('profit_factor', 0):.2f}
Avg Trade Return:   {trades.get('average_trade_return', 0):.2%}
Best Trade:         {trades.get('best_trade', 0):.2%}
Worst Trade:        {trades.get('worst_trade', 0):.2%}

⚡ EFFICIENCY RATIOS
Calmar Ratio:       {ratios.get('calmar_ratio', 0):.2f}
Sterling Ratio:     {ratios.get('sterling_ratio', 0):.2f}
Recovery Factor:    {ratios.get('recovery_factor', 0):.2f}

📊 STATISTICAL METRICS
Skewness:           {risk.get('skewness', 0):.2f}
Kurtosis:           {risk.get('kurtosis', 0):.2f}
Positive Days:      {risk.get('positive_days', 0)}
Negative Days:      {risk.get('negative_days', 0)}

🕐 TIME METRICS
Trading Years:      {returns.get('trading_years', 0):.1f}
Avg Holding Period: {trades.get('average_holding_period', 0):.1f} days
Underwater %:       {drawdown.get('underwater_periods', {}).get('underwater_percentage', 0):.1%}
"""
    
    return summary.strip()


def calculate_benchmark_comparison(strategy_results: Dict, benchmark_results: Dict) -> Dict:
    """
    Compare strategy performance against benchmark.
    
    Args:
        strategy_results (Dict): Strategy performance metrics
        benchmark_results (Dict): Benchmark performance metrics
        
    Returns:
        Dict: Detailed comparison metrics
    """
    comparison = {}
    
    # Return comparison
    strategy_return = strategy_results.get('performance_metrics', {}).get('total_return', 0)
    benchmark_return = benchmark_results.get('total_return', 0)
    
    comparison['return_comparison'] = {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'excess_return': strategy_return - benchmark_return,
        'outperformance': strategy_return > benchmark_return
    }
    
    # Risk comparison
    strategy_vol = strategy_results.get('performance_metrics', {}).get('volatility', 0)
    benchmark_vol = benchmark_results.get('volatility', 0)
    
    comparison['risk_comparison'] = {
        'strategy_volatility': strategy_vol,
        'benchmark_volatility': benchmark_vol,
        'volatility_difference': strategy_vol - benchmark_vol,
        'lower_volatility': strategy_vol < benchmark_vol
    }
    
    # Risk-adjusted comparison
    strategy_sharpe = strategy_results.get('performance_metrics', {}).get('sharpe_ratio', 0)
    benchmark_sharpe = benchmark_results.get('sharpe_ratio', 0)
    
    comparison['risk_adjusted_comparison'] = {
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'sharpe_difference': strategy_sharpe - benchmark_sharpe,
        'better_risk_adjusted': strategy_sharpe > benchmark_sharpe
    }
    
    # Drawdown comparison
    strategy_dd = strategy_results.get('performance_metrics', {}).get('max_drawdown', 0)
    benchmark_dd = benchmark_results.get('max_drawdown', 0)
    
    comparison['drawdown_comparison'] = {
        'strategy_max_dd': strategy_dd,
        'benchmark_max_dd': benchmark_dd,
        'drawdown_difference': strategy_dd - benchmark_dd,
        'lower_drawdown': strategy_dd < benchmark_dd
    }
    
    return comparison


def export_performance_data(performance_report: Dict, trades: List, 
                          portfolio_history: pd.DataFrame, filepath: str):
    """
    Export comprehensive performance data to Excel file.
    
    Args:
        performance_report (Dict): Performance analysis report
        trades (List): List of completed trades
        portfolio_history (pd.DataFrame): Portfolio value history
        filepath (str): Output file path
    """
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Portfolio history
            portfolio_history.to_excel(writer, sheet_name='Portfolio_History', index=True)
            
            # Trade details
            if trades:
                trade_data = []
                for i, trade in enumerate(trades):
                    trade_dict = {
                        'Trade_Number': i + 1,
                        'Entry_Date': trade.entry_date,
                        'Exit_Date': trade.exit_date,
                        'Entry_Price': trade.entry_price,
                        'Exit_Price': trade.exit_price,
                        'Quantity': trade.quantity,
                        'Side': trade.side,
                        'PnL': trade.pnl,
                        'Return_Pct': trade.return_pct,
                        'Duration_Days': trade.duration_days,
                        'Entry_Signal': trade.entry_signal,
                        'Exit_Signal': trade.exit_signal
                    }
                    trade_data.append(trade_dict)
                
                trades_df = pd.DataFrame(trade_data)
                trades_df.to_excel(writer, sheet_name='Trade_Details', index=False)
            
            # Performance summary
            summary_data = []
            
            # Flatten performance report for tabular format
            for category, metrics in performance_report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if not isinstance(value, (dict, list, pd.Series)):
                            summary_data.append({
                                'Category': category,
                                'Metric': metric,
                                'Value': value
                            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Performance_Summary', index=False)
            
        print(f"Performance data exported to: {filepath}")
        
    except Exception as e:
        print(f"Error exporting performance data: {str(e)}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample portfolio data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simulate portfolio performance with some drift and volatility
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
    portfolio_values = [10000]  # Starting value
    
    for ret in returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    portfolio_data = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'daily_return': [0] + list(returns[1:])
    }, index=dates)
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(portfolio_data)
    
    # Generate performance report
    report = analyzer.generate_performance_report()
    
    # Print formatted summary
    summary = format_performance_summary(report)
    print(summary)
    
    # Print specific metrics
    print("\n=== DETAILED RISK METRICS ===")
    for metric, value in report['risk'].items():
        if isinstance(value, (int, float)):
            if 'ratio' in metric or 'factor' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n=== DRAWDOWN ANALYSIS ===")
    dd_metrics = report['drawdown']
    print(f"Number of Drawdown Periods: {dd_metrics['drawdown_periods']}")
    print(f"Longest Drawdown: {dd_metrics['longest_drawdown_days']} days")
    print(f"Average Drawdown Duration: {dd_metrics['average_drawdown_duration']:.1f} days")
    print(f"Time Underwater: {dd_metrics['underwater_periods']['underwater_percentage']:.1%}")