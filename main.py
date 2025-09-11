"""
Main Execution Script for Moving Average Crossover Backtesting

This script orchestrates the entire backtesting process, from data fetching
to performance analysis and visualization.

Usage:
    python main.py
    
Or with custom parameters:
    python main.py --symbol AAPL --start 2020-01-01 --end 2024-01-01 --short_ma 20 --long_ma 50
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
import pandas as pd
import warnings

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import fetch_stock_data
from strategy import MovingAverageCrossoverStrategy
from backtest_engine import BacktestEngine, BacktestConfig, run_buy_and_hold_benchmark
from performance_metrics import PerformanceAnalyzer, format_performance_summary, calculate_benchmark_comparison
from visualization import BacktestVisualizer, save_all_plots

warnings.filterwarnings('ignore')


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file {config_path} not found. Using default parameters.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration parameters."""
    return {
        'strategy': {
            'short_ma_period': 20,
            'long_ma_period': 50,
            'ma_type': 'SMA'
        },
        'backtest': {
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'initial_capital': 10000,
            'position_size': 1.0,
            'transaction_cost': 0.001
        },
        'stock': {
            'symbol': 'AAPL'
        },
        'output': {
            'save_results': True,
            'save_plots': True,
            'results_dir': 'data/'
        }
    }


def create_output_directory(results_dir: str) -> str:
    """
    Create output directory with timestamp.
    
    Args:
        results_dir (str): Base directory for results
        
    Returns:
        str: Full path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(results_dir, f"backtest_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_results_to_csv(results: dict, output_dir: str, symbol: str):
    """
    Save backtest results to CSV files.
    
    Args:
        results (dict): Backtest results
        output_dir (str): Output directory
        symbol (str): Stock symbol
    """
    try:
        # Save portfolio history
        portfolio_df = results['portfolio_history']
        portfolio_path = os.path.join(output_dir, f"{symbol}_portfolio_history.csv")
        portfolio_df.to_csv(portfolio_path)
        print(f"Portfolio history saved to: {portfolio_path}")
        
        # Save trade details
        if results['trades']:
            trades_data = []
            for i, trade in enumerate(results['trades']):
                trade_dict = {
                    'trade_number': i + 1,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'side': trade.side,
                    'pnl': trade.pnl,
                    'return_pct': trade.return_pct,
                    'duration_days': trade.duration_days
                }
                trades_data.append(trade_dict)
            
            trades_df = pd.DataFrame(trades_data)
            trades_path = os.path.join(output_dir, f"{symbol}_trades.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade details saved to: {trades_path}")
        
        # Save performance summary
        performance_data = []
        for category, metrics in results.items():
            if category in ['trade_statistics', 'performance_metrics'] and isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if not isinstance(value, (dict, list)):
                        performance_data.append({
                            'category': category,
                            'metric': metric,
                            'value': value
                        })
        
        if performance_data:
            summary_df = pd.DataFrame(performance_data)
            summary_path = os.path.join(output_dir, f"{symbol}_performance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Performance summary saved to: {summary_path}")
            
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")


def run_backtest(config: dict, args: argparse.Namespace = None) -> dict:
    """
    Run the complete backtesting process.
    
    Args:
        config (dict): Configuration parameters
        args (argparse.Namespace, optional): Command line arguments
        
    Returns:
        dict: Complete backtest results
    """
    # Override config with command line arguments if provided
    if args:
        if args.symbol:
            config['stock']['symbol'] = args.symbol
        if args.start:
            config['backtest']['start_date'] = args.start
        if args.end:
            config['backtest']['end_date'] = args.end
        if args.short_ma:
            config['strategy']['short_ma_period'] = args.short_ma
        if args.long_ma:
            config['strategy']['long_ma_period'] = args.long_ma
        if args.capital:
            config['backtest']['initial_capital'] = args.capital
    
    # Extract parameters
    symbol = config['stock']['symbol']
    start_date = config['backtest']['start_date']
    end_date = config['backtest']['end_date']
    short_ma = config['strategy']['short_ma_period']
    long_ma = config['strategy']['long_ma_period']
    ma_type = config['strategy']['ma_type']
    initial_capital = config['backtest']['initial_capital']
    position_size = config['backtest']['position_size']
    transaction_cost = config['backtest']['transaction_cost']
    
    print("=" * 60)
    print("🚀 MOVING AVERAGE CROSSOVER BACKTESTING")
    print("=" * 60)
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Moving Averages: {short_ma} / {long_ma} ({ma_type})")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Position Size: {position_size:.0%}")
    print(f"Transaction Cost: {transaction_cost:.1%}")
    print("=" * 60)
    
    try:
        # Step 1: Fetch data and calculate moving averages
        print("\n📊 Step 1: Fetching stock data...")
        data = fetch_stock_data(symbol, start_date, end_date, short_ma, long_ma, ma_type)
        print(f"✅ Successfully fetched {len(data)} days of data")
        
        # Step 2: Generate trading signals
        print("\n📈 Step 2: Generating trading signals...")
        strategy = MovingAverageCrossoverStrategy(short_ma, long_ma)
        data_with_signals = strategy.generate_signals(data)
        
        # Validate signals
        is_valid = strategy.validate_signals(data_with_signals)
        if not is_valid:
            print("⚠️  Warning: Signal validation failed")
        else:
            print("✅ Signal validation passed")
        
        # Step 3: Run backtest
        print("\n🔄 Step 3: Running backtest simulation...")
        backtest_config = BacktestConfig(
            initial_capital=initial_capital,
            position_size=position_size,
            transaction_cost=transaction_cost
        )
        
        engine = BacktestEngine(backtest_config)
        backtest_results = engine.run_backtest(data_with_signals)
        print("✅ Backtest simulation completed")
        
        # Step 4: Calculate performance metrics
        print("\n📊 Step 4: Calculating performance metrics...")
        portfolio_data = backtest_results['portfolio_history']
        analyzer = PerformanceAnalyzer(portfolio_data)
        performance_report = analyzer.generate_performance_report(backtest_results['trades'])
        
        # Step 5: Run buy-and-hold benchmark
        print("\n📋 Step 5: Running buy-and-hold benchmark...")
        benchmark_results = run_buy_and_hold_benchmark(data, initial_capital, transaction_cost)
        
        # Step 6: Compare with benchmark
        comparison = calculate_benchmark_comparison(backtest_results, benchmark_results)
        
        # Combine all results
        complete_results = {
            'stock_data': data_with_signals,
            'backtest_results': backtest_results,
            'performance_report': performance_report,
            'benchmark_results': benchmark_results,
            'comparison': comparison,
            'config': config
        }
        
        print("✅ Performance analysis completed")
        return complete_results
        
    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        raise


def display_results(results: dict):
    """
    Display comprehensive backtest results.
    
    Args:
        results (dict): Complete backtest results
    """
    print("\n" + "=" * 80)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Strategy Performance
    backtest_results = results['backtest_results']
    performance_metrics = backtest_results['performance_metrics']
    trade_stats = backtest_results['trade_statistics']
    
    print(f"\n🎯 STRATEGY PERFORMANCE")
    print(f"Total Return:        {performance_metrics['total_return']:.2%}")
    print(f"CAGR:               {performance_metrics['cagr']:.2%}")
    print(f"Volatility:         {performance_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio:       {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:       {performance_metrics['max_drawdown']:.2%}")
    print(f"Calmar Ratio:       {performance_metrics['cagr'] / performance_metrics['max_drawdown']:.2f}" if performance_metrics['max_drawdown'] > 0 else "Calmar Ratio:       ∞")
    
    # Trade Statistics
    print(f"\n📈 TRADE STATISTICS")
    print(f"Total Trades:       {trade_stats['total_trades']}")
    print(f"Win Rate:           {trade_stats['win_rate']:.2%}")
    print(f"Profit Factor:      {trade_stats['profit_factor']:.2f}")
    print(f"Average Return:     {trade_stats['average_return']:.2%}")
    print(f"Best Trade:         {trade_stats['best_trade']:.2%}")
    print(f"Worst Trade:        {trade_stats['worst_trade']:.2%}")
    
    # Benchmark Comparison
    benchmark_results = results['benchmark_results']
    comparison = results['comparison']
    
    print(f"\n📋 BENCHMARK COMPARISON")
    print(f"Strategy Return:    {performance_metrics['total_return']:.2%}")
    print(f"Benchmark Return:   {benchmark_results['total_return']:.2%}")
    print(f"Excess Return:      {comparison['return_comparison']['excess_return']:.2%}")
    print(f"Outperformed:       {'✅ YES' if comparison['return_comparison']['outperformance'] else '❌ NO'}")
    
    # Risk Comparison
    print(f"\n⚠️  RISK COMPARISON")
    print(f"Strategy Volatility: {performance_metrics['volatility']:.2%}")
    print(f"Benchmark Volatility: {benchmark_results['volatility']:.2%}")
    print(f"Strategy Sharpe:    {performance_metrics['sharpe_ratio']:.2f}")
    print(f"Benchmark Sharpe:   {benchmark_results['sharpe_ratio']:.2f}")
    print(f"Better Risk-Adj:    {'✅ YES' if comparison['risk_adjusted_comparison']['better_risk_adjusted'] else '❌ NO'}")


def create_visualizations(results: dict, output_dir: str = None):
    """
    Create comprehensive visualizations for backtest results.
    
    Args:
        results (dict): Complete backtest results
        output_dir (str, optional): Directory to save plots
    """
    print("\n🎨 Step 6: Creating visualizations...")
    
    try:
        # Extract data
        stock_data = results['stock_data']
        backtest_results = results['backtest_results']
        performance_report = results['performance_report']
        benchmark_data = results['stock_data'][['Close']].copy()  # Use price data as benchmark
        config = results['config']
        
        # Initialize visualizer
        visualizer = BacktestVisualizer()
        
        # Create comprehensive report
        figures = visualizer.create_comprehensive_report(
            data=stock_data,
            performance_report=performance_report,
            trades=backtest_results['trades'],
            portfolio_data=backtest_results['portfolio_history'],
            benchmark_data=benchmark_data,
            short_period=config['strategy']['short_ma_period'],
            long_period=config['strategy']['long_ma_period'],
            save_dir=output_dir
        )
        
        # Create additional plots
        print("Creating additional analysis charts...")
        
        # Rolling metrics
        rolling_fig = visualizer.plot_rolling_metrics(
            backtest_results['portfolio_history'],
            save_path=f"{output_dir}/rolling_metrics.png" if output_dir else None
        )
        figures['rolling_metrics'] = rolling_fig
        
        # Correlation analysis
        correlation_fig = visualizer.plot_correlation_analysis(
            backtest_results['portfolio_history'],
            benchmark_data,
            save_path=f"{output_dir}/correlation_analysis.png" if output_dir else None
        )
        figures['correlation_analysis'] = correlation_fig
        
        print(f"✅ Created {len(figures)} visualization charts")
        
        if output_dir:
            print(f"📁 All charts saved to: {output_dir}")
        
        return figures
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        return {}


def generate_text_report(results: dict, output_dir: str = None):
    """
    Generate a comprehensive text report.
    
    Args:
        results (dict): Complete backtest results
        output_dir (str, optional): Directory to save report
    """
    try:
        # Format comprehensive report
        performance_summary = format_performance_summary(results['performance_report'])
        
        # Create detailed report
        report_text = f"""
{'='*80}
MOVING AVERAGE CROSSOVER STRATEGY - BACKTEST REPORT
{'='*80}

CONFIGURATION:
Symbol: {results['config']['stock']['symbol']}
Period: {results['config']['backtest']['start_date']} to {results['config']['backtest']['end_date']}
Moving Averages: {results['config']['strategy']['short_ma_period']} / {results['config']['strategy']['long_ma_period']} ({results['config']['strategy']['ma_type']})
Initial Capital: ${results['config']['backtest']['initial_capital']:,.2f}
Position Size: {results['config']['backtest']['position_size']:.0%}
Transaction Cost: {results['config']['backtest']['transaction_cost']:.1%}

{performance_summary}

BENCHMARK COMPARISON:
{'-'*40}
Strategy Total Return:  {results['backtest_results']['performance_metrics']['total_return']:.2%}
Buy & Hold Return:      {results['benchmark_results']['total_return']:.2%}
Excess Return:          {results['comparison']['return_comparison']['excess_return']:.2%}
Outperformance:         {'YES' if results['comparison']['return_comparison']['outperformance'] else 'NO'}

Strategy Volatility:    {results['backtest_results']['performance_metrics']['volatility']:.2%}
Benchmark Volatility:   {results['benchmark_results']['volatility']:.2%}

Strategy Sharpe:        {results['backtest_results']['performance_metrics']['sharpe_ratio']:.2f}
Benchmark Sharpe:       {results['benchmark_results']['sharpe_ratio']:.2f}

TRADE DETAILS:
{'-'*40}
"""
        
        # Add individual trade details if available
        if results['backtest_results']['trades']:
            report_text += "Individual Trades:\n"
            for i, trade in enumerate(results['backtest_results']['trades'][:10]):  # Show first 10 trades
                report_text += f"{i+1:2d}. {trade.entry_date.strftime('%Y-%m-%d')} to {trade.exit_date.strftime('%Y-%m-%d')}: "
                report_text += f"{trade.return_pct:+.2%} ({trade.duration_days} days)\n"
            
            if len(results['backtest_results']['trades']) > 10:
                report_text += f"... and {len(results['backtest_results']['trades']) - 10} more trades\n"
        
        # Add strategy analysis
        report_text += f"""
STRATEGY ANALYSIS:
{'-'*40}
This moving average crossover strategy used {results['config']['strategy']['short_ma_period']}-day and {results['config']['strategy']['long_ma_period']}-day moving averages
to generate buy and sell signals. The strategy {'outperformed' if results['comparison']['return_comparison']['outperformance'] else 'underperformed'} 
the buy-and-hold benchmark by {results['comparison']['return_comparison']['excess_return']:.2%}.

Key Insights:
- The strategy completed {results['backtest_results']['trade_statistics']['total_trades']} trades with a {results['backtest_results']['trade_statistics']['win_rate']:.1%} win rate
- Maximum drawdown was {results['backtest_results']['performance_metrics']['max_drawdown']:.2%}, {'better' if results['comparison']['drawdown_comparison']['lower_drawdown'] else 'worse'} than benchmark
- Risk-adjusted returns (Sharpe ratio) were {'superior' if results['comparison']['risk_adjusted_comparison']['better_risk_adjusted'] else 'inferior'} to the benchmark
- Average holding period was {results['backtest_results']['trade_statistics']['average_holding_period']:.1f} days

DISCLAIMER:
This backtest is for educational purposes only and does not constitute investment advice.
Past performance does not guarantee future results. Consider transaction costs, slippage,
and market conditions when implementing any trading strategy.

Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        # Save report if output directory specified
        if output_dir:
            report_path = os.path.join(output_dir, f"{results['config']['stock']['symbol']}_backtest_report.txt")
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Comprehensive report saved to: {report_path}")
        
        return report_text
        
    except Exception as e:
        print(f"❌ Error generating text report: {str(e)}")
        return ""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Moving Average Crossover Backtesting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --symbol MSFT --start 2019-01-01 --end 2024-01-01
  python main.py --symbol TSLA --short_ma 10 --long_ma 30 --capital 25000
  python main.py --config custom_config.yaml
        """
    )
    
    parser.add_argument('--symbol', '-s', type=str, 
                       help='Stock ticker symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD format)')
    parser.add_argument('--short_ma', type=int,
                       help='Short-term moving average period')
    parser.add_argument('--long_ma', type=int,
                       help='Long-term moving average period')
    parser.add_argument('--capital', type=float,
                       help='Initial capital amount')
    parser.add_argument('--config', '-c', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = None
    if not args.no_save and config['output']['save_results']:
        output_dir = create_output_directory(config['output']['results_dir'])
        print(f"📁 Results will be saved to: {output_dir}")
    
    try:
        # Run backtest
        results = run_backtest(config, args)
        
        # Display results
        display_results(results)
        
        # Save results to CSV
        if output_dir:
            save_results_to_csv(results['backtest_results'], output_dir, config['stock']['symbol'])
        
        # Generate text report
        text_report = generate_text_report(results, output_dir)
        
        # Create visualizations
        if not args.no_plots:
            figures = create_visualizations(results, output_dir)
            
            # Show plots if not saving
            if not output_dir:
                print("\n🎨 Displaying charts... Close plot windows to continue.")
                import matplotlib.pyplot as plt
                plt.show()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎉 BACKTESTING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        strategy_return = results['backtest_results']['performance_metrics']['total_return']
        benchmark_return = results['benchmark_results']['total_return']
        
        print(f"Strategy Return: {strategy_return:.2%}")
        print(f"Benchmark Return: {benchmark_return:.2%}")
        print(f"Excess Return: {strategy_return - benchmark_return:+.2%}")
        
        if output_dir:
            print(f"\n📁 All results saved to: {output_dir}")
            print("📊 Files generated:")
            print(f"  - Portfolio history CSV")
            print(f"  - Trade details CSV") 
            print(f"  - Performance summary CSV")
            print(f"  - Comprehensive text report")
            if not args.no_plots:
                print(f"  - Visualization charts (PNG)")
        
        print("\n💡 Next steps:")
        print("  - Review the performance metrics and trade analysis")
        print("  - Consider adjusting MA periods or risk parameters")
        print("  - Test on different time periods or stocks")
        print("  - Implement additional risk management rules")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Backtesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()