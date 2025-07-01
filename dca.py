import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import newton
import warnings
warnings.filterwarnings('ignore')

class DCASimulator:
    """
    Dollar Cost Averaging Investment Simulator
    Simulates periodic investments with performance metrics and visualization
    """
    
    def __init__(self, asset_prices, benchmark_prices, investment_amount, frequency='M'):
        """
        Initialize DCA Simulator
        
        Parameters:
        -----------
        asset_prices : pd.Series
            Time series of asset prices (index should be DatetimeIndex)
        benchmark_prices : pd.Series
            Time series of benchmark (money market) prices
        investment_amount : float
            Amount to invest each period
        frequency : str
            Investment frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Y' (yearly)
        """
        self.asset_prices = asset_prices.sort_index()
        self.benchmark_prices = benchmark_prices.sort_index()
        self.investment_amount = investment_amount
        self.frequency = frequency
        
        # Align series to common dates
        common_dates = self.asset_prices.index.intersection(self.benchmark_prices.index)
        self.asset_prices = self.asset_prices[common_dates]
        self.benchmark_prices = self.benchmark_prices[common_dates]
        
        # Calculate daily returns
        self.asset_returns = self.asset_prices.pct_change().fillna(0)
        self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0)
        
        # Initialize results
        self.simulation_results = None
        
    def simulate(self):
        """Run DCA simulation"""
        # Generate investment dates based on frequency
        investment_dates = self._generate_investment_dates()
        investment_dates_set = set(investment_dates)  # Convert to set for faster lookup
        
        # Initialize tracking variables
        units_held = 0
        total_invested = 0
        results = []
        
        for date in self.asset_prices.index:
            # Check if it's an investment date
            if date in investment_dates_set:
                # Buy units
                price = self.asset_prices[date]
                units_bought = self.investment_amount / price
                units_held += units_bought
                total_invested += self.investment_amount
            
            # Calculate market value
            market_value = units_held * self.asset_prices[date]
            
            # Store results
            results.append({
                'date': date,
                'units_held': units_held,
                'total_invested': total_invested,
                'market_value': market_value,
                'asset_price': self.asset_prices[date],
                'benchmark_price': self.benchmark_prices[date]
            })
        
        self.simulation_results = pd.DataFrame(results)
        self.simulation_results.set_index('date', inplace=True)
        
        # Calculate returns
        self._calculate_performance_metrics()
        
        return self.simulation_results
    
    def _generate_investment_dates(self):
        """Generate investment dates based on frequency"""
        start_date = self.asset_prices.index[0]
        end_date = self.asset_prices.index[-1]
        
        if self.frequency == 'D':
            # Daily investments
            investment_dates = self.asset_prices.index
        elif self.frequency == 'W':
            # Weekly investments - find all Mondays in the data
            investment_dates = self.asset_prices.index[self.asset_prices.index.weekday == 0]
        elif self.frequency == 'M':
            # Monthly investments - since your data is monthly, invest on all dates
            investment_dates = self.asset_prices.index
        elif self.frequency == 'Y':
            # Yearly investments - take first date of each year
            investment_dates = self.asset_prices.index[self.asset_prices.index.year.duplicated() == False]
        else:
            raise ValueError(f"Invalid frequency: {self.frequency}")
        
        return investment_dates
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics"""
        df = self.simulation_results
        
        # Calculate IRR (Internal Rate of Return)
        cash_flows = []
        cash_flow_dates = []
        
        # Track investment dates
        df['invested_this_period'] = df['total_invested'].diff().fillna(df['total_invested'].iloc[0])
        
        # Add investment cash flows (negative)
        for i, (date, row) in enumerate(df.iterrows()):
            if row['invested_this_period'] > 0:
                cash_flows.append(-row['invested_this_period'])
                cash_flow_dates.append(date)
        
        # Add final value (positive) - only if we have investments
        if len(cash_flows) > 0 and df['market_value'].iloc[-1] > 0:
            cash_flows.append(df['market_value'].iloc[-1])
            cash_flow_dates.append(df.index[-1])
            
            # Calculate IRR
            try:
                # Use XIRR approach - more accurate for irregular cash flows
                df['IRR_annualized'] = self._calculate_xirr(cash_flows, cash_flow_dates)
            except:
                # Fallback: calculate simple annualized return
                years = (df.index[-1] - df.index[0]).days / 365.25
                df['IRR_annualized'] = (df['market_value'].iloc[-1] / df['total_invested'].iloc[-1]) ** (1/years) - 1
        else:
            df['IRR_annualized'] = 0
        
        # Portfolio cumulative return
        df['portfolio_return'] = df['market_value'] / df['total_invested'] - 1
        df['portfolio_return'] = df['portfolio_return'].fillna(0)
        df['portfolio_value_index'] = 100 * (1 + df['portfolio_return'])
        
        # Benchmark return index
        df['benchmark_return'] = df['benchmark_price'] / df['benchmark_price'].iloc[0] - 1
        df['benchmark_index'] = 100 * (1 + df['benchmark_return'])
        
        # Drawdown calculation
        df['cumulative_max'] = df['market_value'].cummax()
        df['drawdown'] = (df['market_value'] - df['cumulative_max']) / df['cumulative_max']
        
        # Daily returns for volatility calculation
        # Calculate returns based on the portfolio (not just market value changes)
        df['portfolio_value_change'] = df['market_value'] - df['total_invested']
        df['portfolio_daily_return'] = df['portfolio_value_change'].pct_change().fillna(0)
        
        # For proper volatility, we need returns on the actual asset
        df['asset_daily_return'] = self.asset_returns
    
    def _calculate_xirr(self, cash_flows, dates):
        """Calculate XIRR (IRR for irregular cash flows)"""
        # Convert dates to years from first date
        first_date = dates[0]
        years = [(d - first_date).days / 365.25 for d in dates]
        
        # Define NPV function
        def npv(rate):
            return sum(cf / (1 + rate) ** y for cf, y in zip(cash_flows, years))
        
        # Try different initial guesses
        for guess in [0.1, 0.0, -0.1, 0.3, -0.3]:
            try:
                # Use bounded optimization to prevent extreme values
                from scipy.optimize import brentq
                # IRR must be between -99% and 1000% annually
                return brentq(npv, -0.99, 10.0)
            except:
                continue
        
        # If all fail, use simple return
        total_return = cash_flows[-1] / (-sum(cash_flows[:-1]))
        years_held = years[-1]
        return total_return ** (1/years_held) - 1 if years_held > 0 else 0
        
    def get_performance_summary(self):
        """Get summary performance statistics"""
        if self.simulation_results is None:
            raise ValueError("Run simulate() first")
        
        df = self.simulation_results
        
        # Calculate annualized metrics
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        
        # Total return
        total_return = (df['market_value'].iloc[-1] / df['total_invested'].iloc[-1] - 1) * 100
        annualized_return = ((df['market_value'].iloc[-1] / df['total_invested'].iloc[-1]) ** (1/years) - 1) * 100
        
        # IRR (already calculated)
        irr_annualized = df['IRR_annualized'].iloc[-1] * 100 if not np.isnan(df['IRR_annualized'].iloc[-1]) else 0
        
        # Volatility - use asset returns, not portfolio value changes
        # Since data is monthly, annualize monthly volatility
        monthly_returns = self.asset_returns[self.asset_returns.index.isin(df.index)]
        monthly_vol = monthly_returns.std()
        annualized_vol = monthly_vol * np.sqrt(12) * 100  # Monthly to annual
        
        # Max drawdown
        max_drawdown = df['drawdown'].min() * 100
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Benchmark comparison
        benchmark_total_return = (df['benchmark_index'].iloc[-1] - 100)
        
        summary = {
            'Total Invested': df['total_invested'].iloc[-1],
            'Market Value': df['market_value'].iloc[-1],
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'IRR (%)': irr_annualized,
            'Annualized Volatility (%)': annualized_vol,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Benchmark Total Return (%)': benchmark_total_return,
            'Excess Return vs Benchmark (%)': total_return - benchmark_total_return
        }
        
        return summary
    
    def plot_results(self, figsize=(15, 10)):
        """Create comprehensive visualization of results"""
        if self.simulation_results is None:
            raise ValueError("Run simulate() first")
        
        df = self.simulation_results
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('DCA Investment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Investment and Market Value Over Time
        ax1.plot(df.index, df['total_invested'], label='Total Invested', linewidth=2)
        ax1.plot(df.index, df['market_value'], label='Market Value', linewidth=2)
        ax1.fill_between(df.index, df['total_invested'], df['market_value'], 
                        where=df['market_value'] >= df['total_invested'], 
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(df.index, df['total_invested'], df['market_value'], 
                        where=df['market_value'] < df['total_invested'], 
                        color='red', alpha=0.3, label='Loss')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.set_title('Portfolio Value vs Investment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Index Comparison
        ax2.plot(df.index, df['portfolio_value_index'], label='Portfolio Index', linewidth=2)
        ax2.plot(df.index, df['benchmark_index'], label='Benchmark Index', linewidth=2)
        
        # Add IRR line
        if not np.isnan(df['IRR_annualized'].iloc[-1]):
            irr_label = f'IRR: {df["IRR_annualized"].iloc[-1]*100:.2f}%'
            ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax2.text(df.index[len(df)//2], 105, irr_label, fontsize=10, ha='center')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Index Value (Base=100)')
        ax2.set_title('Performance: Portfolio vs Benchmark')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3.fill_between(df.index, df['drawdown'] * 100, 0, color='red', alpha=0.7)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Portfolio Drawdown')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(top=0)
        
        # 4. Rolling Volatility (use asset returns)
        # For monthly data, we can't do 30-day rolling - use 12-month instead
        asset_returns_series = pd.Series(self.asset_returns.values, index=df.index)
        rolling_vol = asset_returns_series.rolling(12, min_periods=3).std() * np.sqrt(12) * 100
        ax4.plot(df.index, rolling_vol, linewidth=2, color='orange')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Annualized Volatility (%)')
        ax4.set_title('12-Month Rolling Volatility')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def export_results(self, filename='dca_results.csv'):
        """Export simulation results to CSV"""
        if self.simulation_results is None:
            raise ValueError("Run simulate() first")
        
        export_df = self.simulation_results[['units_held', 'total_invested', 
                                            'market_value', 'portfolio_value_index', 
                                            'IRR_annualized', 'drawdown', 
                                            'daily_return']].copy()
        export_df['IRR_annualized'] = export_df['IRR_annualized'] * 100  # Convert to percentage
        export_df.to_csv(filename)
        print(f"Results exported to {filename}")


if __name__ == "__main__":
    df = pd.read_csv("data_crypto.csv", sep=';', parse_dates=['Date'], dayfirst=True, decimal=',')
    asset_prices = pd.Series(df['PX_LAST'].values, index=pd.to_datetime(df['Date']))
    
    # Generate benchmark data (money market) for same dates
    # Assuming ~3% annual return for money market
    daily_mm_return = 0.03 / 252  # Convert annual to daily
    benchmark_prices = pd.Series(index=asset_prices.index)
    benchmark_prices.iloc[0] = 100  # Start at 100
    
    for i in range(1, len(benchmark_prices)):
        days_between = (asset_prices.index[i] - asset_prices.index[i-1]).days
        benchmark_prices.iloc[i] = benchmark_prices.iloc[i-1] * (1 + daily_mm_return * days_between)
    
    # Initialize simulator
    simulator = DCASimulator(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        investment_amount=100,  # $1000 per period
        frequency='M'  # Monthly investments
    )
    
    # Run simulation
    results = simulator.simulate()
    
    # Get performance summary
    summary = simulator.get_performance_summary()
    print("\nPerformance Summary:")
    print("-" * 50)
    for metric, value in summary.items():
        print(f"{metric:.<35} {value:>12.2f}")
    
    # Plot results
    fig = simulator.plot_results()
    plt.show()
    
    # Optional: Export results
    # simulator.export_results('crypto_dca_results.csv')