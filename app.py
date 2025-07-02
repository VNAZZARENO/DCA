import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from dca import DCASimulator

# Page configuration
st.set_page_config(
    page_title="DCA Investment Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .performance-metric {
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f4e79;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0.5rem 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class RiskMetrics:
    """Additional risk metrics for comprehensive analysis"""
    
    @staticmethod
    def calculate_var(returns, confidence_level=0.05):
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns, confidence_level=0.05):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0):
        """Calculate Sortino Ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        return excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0

class LumpSumSimulator:
    """Lump Sum investment simulator for comparison"""
    
    def __init__(self, asset_prices, benchmark_prices, lump_sum_amount):
        self.asset_prices = asset_prices.sort_index()
        self.benchmark_prices = benchmark_prices.sort_index()
        self.lump_sum_amount = lump_sum_amount
        
        # Align series
        common_dates = self.asset_prices.index.intersection(self.benchmark_prices.index)
        self.asset_prices = self.asset_prices[common_dates]
        self.benchmark_prices = self.benchmark_prices[common_dates]
        
        self.asset_returns = self.asset_prices.pct_change().fillna(0)
        self.benchmark_returns = self.benchmark_prices.pct_change().fillna(0)
    
    def simulate(self):
        """Simulate lump sum investment"""
        initial_price = self.asset_prices.iloc[0]
        units_held = self.lump_sum_amount / initial_price
        
        results = []
        for date in self.asset_prices.index:
            market_value = units_held * self.asset_prices[date]
            results.append({
                'date': date,
                'market_value': market_value,
                'total_invested': self.lump_sum_amount,
                'asset_price': self.asset_prices[date],
                'benchmark_price': self.benchmark_prices[date]
            })
        
        df = pd.DataFrame(results)
        df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        df['portfolio_return'] = df['market_value'] / df['total_invested'] - 1
        df['portfolio_value_index'] = 100 * (1 + df['portfolio_return'])
        df['benchmark_return'] = df['benchmark_price'] / df['benchmark_price'].iloc[0] - 1
        df['benchmark_index'] = 100 * (1 + df['benchmark_return'])
        df['drawdown'] = (df['market_value'] - df['market_value'].cummax()) / df['market_value'].cummax()
        
        return df

def load_default_data():
    """Load default risk proxies data"""
    try:
        default_path = r"all_risk_proxies.csv"
        description_path = r"data_description.csv"
        
        # Try to load default data
        df = pd.read_csv(default_path, sep=';', decimal=',')
        df['Dates'] = pd.to_datetime(df['Dates'], dayfirst=True)
        df.set_index('Dates', inplace=True)
        
        # Convert all columns except Dates to numeric, handling errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Load descriptions if available
        descriptions = {}
        try:
            desc_df = pd.read_csv(description_path, sep=';')
            
            # Check if the data is in the expected format (rows with Ticker and Description columns)
            if 'Ticker' in desc_df.columns and 'Description' in desc_df.columns:
                # Filter out NaN values properly
                descriptions = {k: v for k, v in dict(zip(desc_df['Ticker'], desc_df['Description'])).items() 
                              if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip()}
            else:
                # The data appears to be transposed - tickers are in first row, descriptions in second row
                # Transpose the dataframe so tickers become column headers
                desc_df_transposed = desc_df.T
                
                # If there are at least 2 rows, use first row as tickers and second as descriptions
                if len(desc_df_transposed) >= 2:
                    tickers = desc_df_transposed.iloc[0].values  # First row contains tickers
                    descriptions_list = desc_df_transposed.iloc[1].values  # Second row contains descriptions
                    
                    # Create the mapping, filtering out empty values
                    descriptions = {}
                    for ticker, desc in zip(tickers, descriptions_list):
                        if pd.notna(ticker) and pd.notna(desc) and str(ticker).strip() and str(desc).strip():
                            descriptions[str(ticker).strip()] = str(desc).strip()
        except Exception as e:
            descriptions = {}
        
        return df, descriptions
    except:
        return None, None

def load_data_from_file(uploaded_file, file_type, sep=';', decimal=',', date_format=None):
    """Load and process data from Excel or CSV file"""
    try:
        if file_type == 'Excel':
            df = pd.read_excel(uploaded_file)
        else:  # CSV
            df = pd.read_csv(uploaded_file, sep=sep, decimal=decimal)
        
        if 'Dates' not in df.columns:
            st.error("File must contain a 'Dates' column")
            return None
        
        # Handle date parsing with custom format
        try:
            if date_format and date_format != 'Auto':
                df['Dates'] = pd.to_datetime(df['Dates'], format=date_format)
            else:
                # Try common date formats
                df['Dates'] = pd.to_datetime(df['Dates'], infer_datetime_format=True, dayfirst=True)
        except:
            try:
                # Fallback to standard parsing
                df['Dates'] = pd.to_datetime(df['Dates'])
            except:
                st.error("Could not parse dates. Please check your date format.")
                return None
        
        df.set_index('Dates', inplace=True)
        
        # Convert all columns except Dates to numeric, handling errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def filter_data_by_date_range(data, date_range, custom_start=None, custom_end=None):
    """Filter data based on selected date range"""
    if data is None or len(data) == 0:
        return data
    
    end_date = data.index.max()
    
    if date_range == "All Time":
        return data
    elif date_range == "YTD":
        start_date = pd.Timestamp(end_date.year, 1, 1)
    elif date_range == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif date_range == "2Y":
        start_date = end_date - pd.DateOffset(years=2)
    elif date_range == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif date_range == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    elif date_range == "10Y":
        start_date = end_date - pd.DateOffset(years=10)
    elif date_range == "Custom":
        start_date = custom_start
        end_date = custom_end
    else:
        return data
    
    # Filter data
    mask = (data.index >= start_date) & (data.index <= end_date)
    return data[mask]

def get_crisis_periods():
    """Define major crisis periods for analysis"""
    return {
        "2008 Financial Crisis": ("2007-07-01", "2009-06-30"),
        "European Debt Crisis": ("2010-01-01", "2012-12-31"),
        "COVID-19 Pandemic": ("2020-01-01", "2021-12-31"),
        "Dot-com Bubble": ("2000-01-01", "2002-12-31"),
        "Russia-Ukraine War": ("2022-02-01", "2023-12-31"),
        "Asian Financial Crisis": ("1997-07-01", "1998-12-31"),
        "Black Monday 1987": ("1987-10-01", "1987-12-31"),
        "Brexit Referendum": ("2016-06-01", "2016-12-31"),
        "Greek Debt Crisis": ("2010-04-01", "2012-12-31"),
        "Oil Crisis 2014-2016": ("2014-06-01", "2016-06-30")
    }

def create_benchmark_series(dates, benchmark_type, benchmark_value, benchmark_data=None):
    """Create benchmark price series"""
    if benchmark_type == "Fixed Yield":
        # Create fixed yield benchmark
        daily_return = benchmark_value / 100 / 252
        benchmark_prices = pd.Series(index=dates, dtype=float)
        benchmark_prices.iloc[0] = 100
        
        for i in range(1, len(benchmark_prices)):
            days_diff = (dates[i] - dates[i-1]).days
            benchmark_prices.iloc[i] = benchmark_prices.iloc[i-1] * (1 + daily_return * days_diff)
        
        return benchmark_prices
    
    elif benchmark_type == "Other Index" and benchmark_data is not None:
        return benchmark_data
    
    else:
        # Default to 3% annual return
        daily_return = 0.03 / 252
        benchmark_prices = pd.Series(index=dates, dtype=float)
        benchmark_prices.iloc[0] = 100
        
        for i in range(1, len(benchmark_prices)):
            days_diff = (dates[i] - dates[i-1]).days
            benchmark_prices.iloc[i] = benchmark_prices.iloc[i-1] * (1 + daily_return * days_diff)
        
        return benchmark_prices

def get_hardcoded_descriptions():
    """Hardcoded asset descriptions as fallback"""
    return {
        'PERGLFI FP Equity': 'Pergam Global Fund is a SICAV incorporated in France. The Fund\'s objective is to generate capital gains over the long term. The Fund invests in a diversified portfolio of equity securities, fixed-income instruments, UCITS/AIFs and money market instruments.',
        'H15T3M Index': 'US Treasury Yield Curve Rate T Note Constant Maturity 3 Month',
        'H15T1Y Index': 'US Treasury Yield Curve Rate T Note Constant Maturity 1Y',
        'H15T5Y Index': 'US Treasury Yield Curve Rate T Note Constant Maturity 5Y',
        'H15T10Y Index': 'US Treasury Yield Curve Rate T Note Constant Maturity 10Y',
        'ESES Index': 'S&P500 MINI SPRD M5-U5',
        'OATA Comdty': 'French Gov Active Contract',
        'VGA Index': 'EURO STOXX 50 Active Contract',
        'EURUSD Curncy': 'Euro/US Dollar Exchange Rate',
        'NZDUSD Curncy': 'New Zealand Dollar/US Dollar Exchange Rate',
        'AUDUSD Curncy': 'Australian Dollar/US Dollar Exchange Rate',
        'EURGBP Curncy': 'Euro/British Pound Exchange Rate',
        'CHFUSD Curncy': 'Swiss Franc/US Dollar Exchange Rate',
        'CHFEUR Curncy': 'Swiss Franc/Euro Exchange Rate',
        'USDJPY Curncy': 'US Dollar/Japanese Yen Exchange Rate',
        'EURJPY Curncy': 'Euro/Japanese Yen Exchange Rate',
        'EURCHF Index': 'Euro/Swiss Franc Exchange Rate',
        'XBTUSD Curncy': 'Bitcoin/US Dollar Exchange Rate',
        'XAU Curncy': 'Gold',
        'ERA Comdty': 'Euribor 3Mo',
        'FRANCE CDS USD SR 10Y D14 Corp': '10Y French CDS',
        'UK CDS USD SR 10Y D14 Corp': '10Y UK CDS',
        'SPAIN CDS USD SR 10Y D14 Corp': '10Y Spanish CDS',
        'ITALY CDS USD SR 10Y D14 Corp': '10Y Italian CDS',
        'GERMAN CDS USD SR 10Y D14 Corp': '10 German CDS',
        'GTDEM10Y Govt': 'Generic 10Y German Bond',
        'CTDEM10Y Govt': 'Current 10Y Gov Bond',
        'GDP CURY Index': 'USD GDP Norminal Dollars YoY',
        'CTFRF10Y Govt': 'Current 10Y France Gov Bond',
        'CTDEM10Y Govt': 'Current 10Y German Gov Bond',
        'CTITL10Y Govt': 'Current 10Y Italian Gov Bond',
        'CTEUR10Y Govt': 'Current 10Y Eurozone Gov Bond',
        'GUKG30 Index': 'Current 10Y UK Gov Bond',
        'CTEUR7Y Govt': 'Current 7Y Eurozone Gov Bond',
        'JPEI3MEU Index': 'JPM ESG Global HY Corporate Custom Maturity Index Unhedged in EUR',
        'LF98TRUU Index': 'The Bloomberg US Corporate High Yield Bond Index measures the USD-denominated, high yield, fixed-rate corporate bond market. Securities are classified as high yield if the middle rating of Moody\'s, Fitch and S&P is Ba1/BB+/BB+ or below.',
        'IBXXCHF3 Index': 'Markit iBoxx USD Liquid Investment Grade CHF Unhedged TRI',
        'CLA Comdty': 'Crude Oil Active Contract',
        'ASDA Index': 'S&P500 Active Dividend Future',
        'DEDA Index': 'Active Dividend Dividendes cash bruts ordinaires annoncés et payés ppour chacun',
        'VIX Index': 'VIX Index',
        'VDAX Index': 'VDAX Index',
        'VCAC Index': 'CAC 40 Volatility Index',
        'Move Index': 'The MOVE Index measures U.S. bond market volatility by tracking a basket of OTC options on U.S. interest rate swaps. The Index tracks implied normal yield volatility of a yield curve weighted basket of at-the-money one month options on the 2-year, 5-year, 10-year, and 30-year constant maturity interest rate swaps.',
        'V2X Index': 'Euro Stoxx 50 Volatility Index VSTOXX',
        'BCOM Index': 'Bloomberg Commodity Index (BCOM) est calculé sur une base de rendement excédentaire et reflète les variations de prix des contrats à terme sur matières premières. L\'indice est rééquilibré chaque année selon la pondération (2/3 le volume de négociation et 1/3 capitalisation boursière mondiale)',
        'VG1 Index': 'Euro Stoxx 50',
        'ITRX EUR CDSI GEN 5Y Corp': 'The Markit iTraxx Europe index comprises 125 equally weighted credit default swaps on investment grade European corporate entities, distributed among 4 sub-indices: Financials (Senior & Subordinated), Non-Financials and HiVol.',
        'ITRX XOVER CDSI GEN 5Y Corp': 'The Markit iTraxx Europe Crossover index comprises 75 equally weighted credit default swaps on the most liquid sub-investment grade European corporate entities.',
        'ITRX JAPAN CDSI GEN 5Y Corp': 'The Markit iTraxx Japan index comprises 40 equally-weighted CDS on investment grade Japanese entities.',
        'ITRX EXJP IG CDSI GEN 5Y Corp': 'The Markit iTraxx Asia ex-Japan Investment Grade index comprises 40 equally-weighted investment grade CDS index of Asian entities.',
        'HGK5 Index': 'Copper Future',
        'SIK5 Index': 'Silver Future',
        'XPT BGN Curncy': 'The per Troy ounce spot price for Platinum, in plate or ingot form, with a minimum purity of 99.95%.',
        'ITRX EXJP IG CDSI S42 5Y Corp': 'The Markit iTraxx Asia ex-Japan Investment Grade index comprises 40 equally-weighted investment grade CDS index of Asian entities.',
        'USYC2Y10 Index': 'Selling 2Y and buying 10Y US Treasury',
        'CDX IG CDSI GEN 5Y Corp': 'The Markit CDX North America Investment Grade Index is composed of 125 equally weighted credit default swaps on investment grade entities, distributed among 6 sub-indices: High Volatility,Consumer, Energy, Financial, Industrial, and Technology, Media & Tele-communications.',
        'CDX HY CDSI GEN 5Y Corp': 'Markit CDX North America High Yield Index is composed of 100 non-investment grade entities, distributed among 2 sub-indices: B, BB. All entities are domiciled in North America.'
    }

def categorize_assets(asset_columns, descriptions):
    """Categorize assets by type for better organization"""
    categories = {
        "Pergam Funds": [],
        "US Treasury Rates": [],
        "Government Bonds": [],
        "Equity Indices": [],
        "Currencies (FX)": [],
        "Commodities": [],
        "Credit Default Swaps (CDS)": [],
        "Volatility Indices": [],
        "Corporate Bonds": [],
        "Other": []
    }
    
    # Get hardcoded descriptions as fallback
    hardcoded_descriptions = get_hardcoded_descriptions()
    all_descriptions = {**hardcoded_descriptions, **descriptions}  # descriptions override hardcoded
    
    for asset in asset_columns:
        asset_lower = asset.lower()
        desc = all_descriptions.get(asset, "").lower()
        
        # Pergam Funds
        if 'perglfi' in asset_lower or 'pergam' in desc:
            categories["Pergam Funds"].append(asset)
        
        # US Treasury Rates
        elif any(x in asset_lower for x in ['h15t', 'usyc']):
            categories["US Treasury Rates"].append(asset)
        
        # Government Bonds
        elif any(x in asset_lower for x in ['govt', 'ctdem', 'ctfrf', 'ctitl', 'cteur', 'gtdem', 'gukg', 'gdp cury']):
            categories["Government Bonds"].append(asset)
        
        # Equity Indices & Futures
        elif any(x in asset_lower for x in ['eses', 'vga', 'asda', 'deda', 'vg1', 'index']) and not any(x in asset_lower for x in ['vix', 'vdax', 'vcac', 'move', 'v2x']):
            categories["Equity Indices"].append(asset)
        
        # Currencies (FX)
        elif any(x in asset_lower for x in ['curncy', 'eurusd', 'nzdusd', 'audusd', 'eurgbp', 'chfusd', 'chfeur', 'usdjpy', 'eurjpy', 'eurchf', 'xbtusd']):
            categories["Currencies (FX)"].append(asset)
        
        # Commodities
        elif any(x in asset_lower for x in ['comdty', 'xau', 'xpt', 'era', 'oata', 'cla', 'hgk5', 'sik5', 'bcom']):
            categories["Commodities"].append(asset)
        
        # Credit Default Swaps (CDS)
        elif any(x in asset_lower for x in ['cds', 'cdx', 'itrx']) or 'cds' in desc:
            categories["Credit Default Swaps (CDS)"].append(asset)
        
        # Volatility Indices
        elif any(x in asset_lower for x in ['vix', 'vdax', 'vcac', 'move', 'v2x']):
            categories["Volatility Indices"].append(asset)
        
        # Corporate Bonds
        elif any(x in asset_lower for x in ['jpei3meu', 'lf98truu', 'ibxxchf3']) or 'corporate' in desc or 'bond' in desc:
            categories["Corporate Bonds"].append(asset)
        
        else:
            categories["Other"].append(asset)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def create_relative_performance_heatmaps(dca_simulator, lump_sum_results=None):
    """Create heatmap visualizations for relative performance analysis"""
    
    # Get monthly relative performance data
    heatmap_data = dca_simulator.get_monthly_relative_performance_matrix(lump_sum_results)
    
    years = heatmap_data['years']
    months = heatmap_data['months']
    
    # Create subplots
    if lump_sum_results is not None:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('DCA vs Benchmark - Monthly Relative Performance (%)', 
                          'DCA vs Lump Sum - Monthly Relative Performance (%)'),
            vertical_spacing=0.15
        )
        
        # DCA vs Benchmark heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data['benchmark_matrix'],
                x=months,
                y=[str(year) for year in years],
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="Relative Return (%)", x=1.02, len=0.4, y=0.75),
                hoverongaps=False,
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Relative Performance: %{z:.2f}%<extra></extra>',
                showscale=True
            ),
            row=1, col=1
        )
        
        # DCA vs Lump Sum heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data['lumpsum_matrix'],
                x=months,
                y=[str(year) for year in years],
                colorscale='RdYlGn',
                zmid=0,
                colorbar=dict(title="Relative Return (%)", x=1.02, len=0.4, y=0.25),
                hoverongaps=False,
                hovertemplate='Year: %{y}<br>Month: %{x}<br>Relative Performance: %{z:.2f}%<extra></extra>',
                showscale=True
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Monthly Relative Performance Analysis",
            title_x=0.5
        )
    else:
        # Only benchmark comparison
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data['benchmark_matrix'],
            x=months,
            y=[str(year) for year in years],
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title="Relative Return (%)"),
            hoverongaps=False,
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Relative Performance: %{z:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='DCA vs Benchmark - Monthly Relative Performance (%)',
            height=600,
            title_x=0.5
        )
    
    # Update axes
    fig.update_xaxes(title_text="Month", side="bottom")
    fig.update_yaxes(title_text="Year")
    
    return fig

def create_rolling_relative_performance_chart(dca_simulator, lump_sum_results=None, window=12):
    """Create rolling relative performance chart"""
    
    rel_perf_data = dca_simulator.calculate_relative_performance(lump_sum_results)
    
    # Calculate rolling relative performance
    rel_perf_data['rolling_rel_benchmark'] = rel_perf_data['relative_vs_benchmark'].rolling(window=window).mean() * 100
    
    fig = go.Figure()
    
    # Add benchmark comparison
    fig.add_trace(go.Scatter(
        x=rel_perf_data.index,
        y=rel_perf_data['rolling_rel_benchmark'],
        mode='lines',
        name=f'{window}-Month Rolling Relative Performance vs Benchmark',
        line=dict(color='blue', width=2)
    ))
    
    # Add lump sum comparison if available
    if lump_sum_results is not None and 'relative_vs_lumpsum' in rel_perf_data.columns:
        rel_perf_data['rolling_rel_lumpsum'] = rel_perf_data['relative_vs_lumpsum'].rolling(window=window).mean() * 100
        fig.add_trace(go.Scatter(
            x=rel_perf_data.index,
            y=rel_perf_data['rolling_rel_lumpsum'],
            mode='lines',
            name=f'{window}-Month Rolling Relative Performance vs Lump Sum',
            line=dict(color='orange', width=2)
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig.update_layout(
        title=f'{window}-Month Rolling Relative Performance Analysis',
        xaxis_title='Date',
        yaxis_title='Relative Performance (%)',
        height=500,
        showlegend=True
    )
    
    return fig

def create_annual_relative_performance_summary(dca_simulator, lump_sum_results=None):
    """Create annual relative performance summary table and chart"""
    
    rel_perf_data = dca_simulator.calculate_relative_performance(lump_sum_results)
    
    # Group by year and calculate annual relative performance
    annual_stats = rel_perf_data.groupby('year').agg({
        'relative_vs_benchmark': ['mean', 'std', 'min', 'max'],
        'dca_monthly_return': ['mean', 'std'],
        'benchmark_monthly_return': ['mean', 'std']
    }).round(4) * 100  # Convert to percentage
    
    # Flatten column names
    annual_stats.columns = ['_'.join(col).strip() for col in annual_stats.columns]
    annual_stats.columns = [
        'Rel_Benchmark_Mean', 'Rel_Benchmark_Std', 'Rel_Benchmark_Min', 'Rel_Benchmark_Max',
        'DCA_Return_Mean', 'DCA_Return_Std', 'Benchmark_Return_Mean', 'Benchmark_Return_Std'
    ]
    
    # Add lump sum comparison if available
    if lump_sum_results is not None and 'relative_vs_lumpsum' in rel_perf_data.columns:
        lumpsum_stats = rel_perf_data.groupby('year').agg({
            'relative_vs_lumpsum': ['mean', 'std', 'min', 'max']
        }).round(4) * 100
        
        lumpsum_stats.columns = [
            'Rel_LumpSum_Mean', 'Rel_LumpSum_Std', 'Rel_LumpSum_Min', 'Rel_LumpSum_Max'
        ]
        
        annual_stats = pd.concat([annual_stats, lumpsum_stats], axis=1)
    
    return annual_stats

def create_performance_distribution_chart(dca_simulator, lump_sum_results=None):
    """Create distribution chart of monthly relative performance"""
    
    rel_perf_data = dca_simulator.calculate_relative_performance(lump_sum_results)
    
    fig = make_subplots(
        rows=1, cols=2 if lump_sum_results is not None else 1,
        subplot_titles=['DCA vs Benchmark Distribution', 'DCA vs Lump Sum Distribution'] if lump_sum_results is not None else ['DCA vs Benchmark Distribution']
    )
    
    # Benchmark distribution
    benchmark_rel = rel_perf_data['relative_vs_benchmark'].dropna() * 100
    fig.add_trace(
        go.Histogram(
            x=benchmark_rel,
            nbinsx=30,
            name='Relative Performance vs Benchmark',
            opacity=0.7,
            marker_color='blue'
        ),
        row=1, col=1
    )
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Lump sum distribution if available
    if lump_sum_results is not None and 'relative_vs_lumpsum' in rel_perf_data.columns:
        lumpsum_rel = rel_perf_data['relative_vs_lumpsum'].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=lumpsum_rel,
                nbinsx=30,
                name='Relative Performance vs Lump Sum',
                opacity=0.7,
                marker_color='orange'
            ),
            row=1, col=2
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    
    fig.update_layout(
        title='Monthly Relative Performance Distribution',
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text="Relative Performance (%)")
    fig.update_yaxes(title_text="Frequency")
    
    return fig

def create_win_loss_analysis(dca_simulator, lump_sum_results=None):
    """Create win/loss analysis visualization"""
    
    rel_perf_data = dca_simulator.calculate_relative_performance(lump_sum_results)
    
    # Benchmark analysis
    benchmark_wins = (rel_perf_data['relative_vs_benchmark'] > 0).sum()
    benchmark_losses = (rel_perf_data['relative_vs_benchmark'] <= 0).sum()
    total_periods = len(rel_perf_data['relative_vs_benchmark'].dropna())
    
    data = {
        'Strategy': ['DCA vs Benchmark'],
        'Win Rate (%)': [benchmark_wins / total_periods * 100],
        'Wins': [benchmark_wins],
        'Losses': [benchmark_losses],
        'Avg Win (%)': [rel_perf_data[rel_perf_data['relative_vs_benchmark'] > 0]['relative_vs_benchmark'].mean() * 100],
        'Avg Loss (%)': [rel_perf_data[rel_perf_data['relative_vs_benchmark'] <= 0]['relative_vs_benchmark'].mean() * 100]
    }
    
    # Add lump sum analysis if available
    if lump_sum_results is not None and 'relative_vs_lumpsum' in rel_perf_data.columns:
        lumpsum_wins = (rel_perf_data['relative_vs_lumpsum'] > 0).sum()
        lumpsum_losses = (rel_perf_data['relative_vs_lumpsum'] <= 0).sum()
        
        data['Strategy'].append('DCA vs Lump Sum')
        data['Win Rate (%)'].append(lumpsum_wins / total_periods * 100)
        data['Wins'].append(lumpsum_wins)
        data['Losses'].append(lumpsum_losses)
        data['Avg Win (%)'].append(rel_perf_data[rel_perf_data['relative_vs_lumpsum'] > 0]['relative_vs_lumpsum'].mean() * 100)
        data['Avg Loss (%)'].append(rel_perf_data[rel_perf_data['relative_vs_lumpsum'] <= 0]['relative_vs_lumpsum'].mean() * 100)
    
    return pd.DataFrame(data)

def main():
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Try to load the main logo with company name
        try:
            main_logo_path = "Logo-Pergam-noir-fond-blanc.png"
            st.image(main_logo_path, width=400)
        except:
            # Fallback if logo not found
            st.markdown('<h1 class="main-header">💰 DCA Investment Simulator</h1>', unsafe_allow_html=True)
    
    st.markdown("### PERGAM TOOL for Dollar Cost Average vs Benchmark vs Lump Sum")
    
    # Sidebar with small logo
    try:
        small_logo_path = "pergam_finance_logo.jpg"
        st.sidebar.image(small_logo_path, width=150)
    except:
        pass  # Continue without logo if not found
    
    st.sidebar.markdown('<h2 class="sidebar-header">📊 Configuration</h2>', unsafe_allow_html=True)
    
    # Data Source Selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.selectbox("Data Source", ["Default Risk Proxies", "Upload Custom File", "File Path Input"])
    
    data = None
    descriptions = {}
    
    if data_source == "Default Risk Proxies":
        # Try to load default data
        data, descriptions = load_default_data()
        if data is None:
            st.error("⚠️ Could not load default risk proxies data. Please check the file path or upload a custom file.")
            st.sidebar.markdown("**Fallback to file upload:**")
            data_source = "Upload Custom File"
    
    if data_source == "File Path Input":
        # File path input option
        st.sidebar.subheader("📂 File Path Configuration")
        
        # Main data file path
        file_path = st.sidebar.text_input(
            "Data File Path", 
            placeholder=r"C:\path\to\your\data.csv",
            help="Enter the full path to your data file"
        )
        
        # Optional description file path
        desc_file_path = st.sidebar.text_input(
            "Description File Path (Optional)", 
            placeholder=r"C:\path\to\your\descriptions.csv",
            help="Optional: Enter path to description file"
        )
        
        # File type selection
        file_type = st.sidebar.selectbox("File Type", ["CSV", "Excel"])
        
        # CSV specific parameters
        csv_params = {}
        if file_type == "CSV":
            st.sidebar.markdown("**CSV Parameters:**")
            csv_params['sep'] = st.sidebar.selectbox("Separator", [';', ',', '\\t', '|'], index=0)
            csv_params['decimal'] = st.sidebar.selectbox("Decimal separator", ['.', ','], index=0)
            
            # Date format options
            date_format_options = [
                'Auto',
                '%Y-%m-%d',      # 2023-12-31
                '%d/%m/%Y',      # 31/12/2023
                '%m/%d/%Y',      # 12/31/2023
                '%d-%m-%Y',      # 31-12-2023
                '%d.%m.%Y',      # 31.12.2023
                '%Y%m%d',        # 20231231
                'Custom'
            ]
            csv_params['date_format'] = st.sidebar.selectbox("Date Format", date_format_options)
            
            if csv_params['date_format'] == 'Custom':
                csv_params['date_format'] = st.sidebar.text_input(
                    "Custom Date Format", 
                    placeholder="e.g., %d/%m/%Y %H:%M:%S",
                    help="Use Python strftime format codes"
                )
        
        if not file_path:
            st.info("Please enter a file path to load data.")
            st.markdown("""
            ### Expected File Format:
            - **Dates column**: Must be named 'Dates' with date values
            - **Asset columns**: Each column represents a different asset/index
            
            ### Example paths:
            - Windows: `C:\\data\\my_data.csv`
            - Unix/Mac: `/home/user/data/my_data.csv`
            """)
            return
        
        # Try to load data from file path
        try:
            import os
            if not os.path.exists(file_path):
                st.error(f"❌ File not found: {file_path}")
                return
            
            if file_type == "Excel":
                data = pd.read_excel(file_path)
            else:  # CSV
                data = pd.read_csv(
                    file_path, 
                    sep=csv_params['sep'], 
                    decimal=csv_params['decimal']
                )
            
            if 'Dates' not in data.columns:
                st.error("File must contain a 'Dates' column")
                return
            
            # Handle date parsing
            try:
                if csv_params.get('date_format') and csv_params['date_format'] != 'Auto':
                    data['Dates'] = pd.to_datetime(data['Dates'], format=csv_params['date_format'])
                else:
                    data['Dates'] = pd.to_datetime(data['Dates'], infer_datetime_format=True, dayfirst=True)
            except:
                try:
                    data['Dates'] = pd.to_datetime(data['Dates'])
                except:
                    st.error("Could not parse dates. Please check your date format.")
                    return
            
            data.set_index('Dates', inplace=True)
            
            # Convert all columns to numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            st.sidebar.success(f"✅ Loaded data from: {file_path}")
            st.sidebar.info(f"📊 Shape: {data.shape[0]} rows × {data.shape[1]} columns")
            
            # Load descriptions from file path if provided
            if desc_file_path and os.path.exists(desc_file_path):
                try:
                    desc_df = pd.read_csv(desc_file_path, sep=';')
                    
                    if 'Ticker' in desc_df.columns and 'Description' in desc_df.columns:
                        descriptions = {k: v for k, v in dict(zip(desc_df['Ticker'], desc_df['Description'])).items() 
                                      if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip()}
                    else:
                        # Transposed format
                        desc_df_transposed = desc_df.T
                        if len(desc_df_transposed) >= 2:
                            tickers = desc_df_transposed.iloc[0].values
                            descriptions_list = desc_df_transposed.iloc[1].values
                            descriptions = {}
                            for ticker, desc in zip(tickers, descriptions_list):
                                if pd.notna(ticker) and pd.notna(desc) and str(ticker).strip() and str(desc).strip():
                                    descriptions[str(ticker).strip()] = str(desc).strip()
                    
                    st.sidebar.success(f"✅ Loaded {len(descriptions)} descriptions")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading descriptions: {str(e)}")
                    descriptions = {}
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    elif data_source == "Upload Custom File" or data is None:
        # File type selection
        file_type = st.sidebar.selectbox("File Type", ["Excel", "CSV"])
        
        if file_type == "Excel":
            uploaded_file = st.sidebar.file_uploader("Upload Excel file with asset data", type=['xlsx', 'xls'])
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV file with asset data", type=['csv'])
        
        # Optional description file upload
        st.sidebar.markdown("**Optional: Upload description file**")
        description_file = st.sidebar.file_uploader("Upload description file (CSV with Ticker and Description columns)", type=['csv'], key="desc_file")
        
        # CSV specific parameters
        csv_params = {}
        if file_type == "CSV":
            st.sidebar.markdown("**CSV Parameters:**")
            csv_params['sep'] = st.sidebar.selectbox("Separator", [';', ',', '\t', '|'], index=0)
            csv_params['decimal'] = st.sidebar.selectbox("Decimal separator", ['.', ','], index=0)
            
            # Date format options
            date_format_options = [
                'Auto',
                '%Y-%m-%d',      # 2023-12-31
                '%d/%m/%Y',      # 31/12/2023
                '%m/%d/%Y',      # 12/31/2023
                '%d-%m-%Y',      # 31-12-2023
                '%d.%m.%Y',      # 31.12.2023
                '%Y%m%d',        # 20231231
                'Custom'
            ]
            csv_params['date_format'] = st.sidebar.selectbox("Date Format", date_format_options)
            
            if csv_params['date_format'] == 'Custom':
                csv_params['date_format'] = st.sidebar.text_input(
                    "Custom Date Format", 
                    placeholder="e.g., %d/%m/%Y %H:%M:%S",
                    help="Use Python strftime format codes"
                )
        
        if uploaded_file is None:
            st.info(f"Please upload a {file_type.lower()} file with your asset data to begin analysis.")
            st.markdown(f"""
            ### Expected {file_type} Format:
            - **Dates column**: Must be named 'Dates' with date values
            - **Asset columns**: Each column represents a different asset/index
            - **Rich dataset available**: The default dataset includes major indices, commodities, currencies, bonds, CDS, volatility indices, and more
            
            {'**CSV Parameters**: Configure separator, decimal, and date format above' if file_type == 'CSV' else ''}
            """)
            return
        
        # Load data
        if file_type == "Excel":
            data = load_data_from_file(uploaded_file, file_type)
        else:
            data = load_data_from_file(
                uploaded_file, 
                file_type, 
                sep=csv_params['sep'], 
                decimal=csv_params['decimal'], 
                date_format=csv_params['date_format']
            )
        
        if data is None:
            return
        
        # Load descriptions from uploaded file if provided
        if description_file is not None:
            try:
                desc_df = pd.read_csv(description_file, sep=';')
                
                # Check if the data is in the expected format (rows with Ticker and Description columns)
                if 'Ticker' in desc_df.columns and 'Description' in desc_df.columns:
                    # Filter out NaN values properly
                    descriptions = {k: v for k, v in dict(zip(desc_df['Ticker'], desc_df['Description'])).items() 
                                  if pd.notna(k) and pd.notna(v) and str(k).strip() and str(v).strip()}
                else:
                    # The data appears to be transposed - tickers are in first row, descriptions in second row
                    # Transpose the dataframe so tickers become column headers
                    desc_df_transposed = desc_df.T
                    
                    # If there are at least 2 rows, use first row as tickers and second as descriptions
                    if len(desc_df_transposed) >= 2:
                        tickers = desc_df_transposed.iloc[0].values  # First row contains tickers
                        descriptions_list = desc_df_transposed.iloc[1].values  # Second row contains descriptions
                        
                        # Create the mapping, filtering out empty values
                        descriptions = {}
                        for ticker, desc in zip(tickers, descriptions_list):
                            if pd.notna(ticker) and pd.notna(desc) and str(ticker).strip() and str(desc).strip():
                                descriptions[str(ticker).strip()] = str(desc).strip()
                
                st.sidebar.success(f"✅ Loaded {len(descriptions)} asset descriptions")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading description file: {str(e)}")
                descriptions = {}
    
    # Date Range Selection
    st.sidebar.subheader("📅 Date Range")
    date_range_options = ["All Time", "YTD", "1Y", "2Y", "3Y", "5Y", "10Y", "Custom", "Crisis Period"]
    date_range = st.sidebar.selectbox("Select Date Range", date_range_options)
    
    custom_start = None
    custom_end = None
    crisis_start = None
    crisis_end = None
    
    if date_range == "Custom":
        st.sidebar.markdown("**Custom Date Range:**")
        custom_start = st.sidebar.date_input("Start Date", value=data.index.min().date())
        custom_end = st.sidebar.date_input("End Date", value=data.index.max().date())
        custom_start = pd.Timestamp(custom_start)
        custom_end = pd.Timestamp(custom_end)
    
    elif date_range == "Crisis Period":
        crisis_periods = get_crisis_periods()
        crisis_names = list(crisis_periods.keys())
        selected_crisis = st.sidebar.selectbox("Select Crisis Period", crisis_names)
        
        crisis_start_str, crisis_end_str = crisis_periods[selected_crisis]
        crisis_start = pd.Timestamp(crisis_start_str)
        crisis_end = pd.Timestamp(crisis_end_str)
        
        # Check if crisis period has data
        crisis_mask = (data.index >= crisis_start) & (data.index <= crisis_end)
        if not crisis_mask.any():
            available_start = data.index.min().strftime('%Y-%m-%d')
            available_end = data.index.max().strftime('%Y-%m-%d')
            st.sidebar.warning(f"⚠️ No data available for {selected_crisis} ({crisis_start_str} to {crisis_end_str}). Available data: {available_start} to {available_end}")
        else:
            crisis_data_points = crisis_mask.sum()
            st.sidebar.info(f"📊 {crisis_data_points} data points available for {selected_crisis}")
    
    # Apply date filtering
    if date_range == "Crisis Period":
        filtered_data = filter_data_by_date_range(data, "Custom", crisis_start, crisis_end)
    elif date_range == "Custom":
        filtered_data = filter_data_by_date_range(data, date_range, custom_start, custom_end)
    else:
        filtered_data = filter_data_by_date_range(data, date_range)
    
    if filtered_data is None or len(filtered_data) == 0:
        st.error("❌ No data available for the selected date range. Please choose a different range.")
        return
    
    # Show date range info
    st.sidebar.success(f"📊 Data range: {filtered_data.index.min().strftime('%Y-%m-%d')} to {filtered_data.index.max().strftime('%Y-%m-%d')}")
    st.sidebar.info(f"📈 {len(filtered_data)} data points")
    
    # Move asset selection to main area - we'll add it after date range selection
    asset_columns = [col for col in filtered_data.columns if col != 'Dates']
    
    # ASSET SELECTION IN MAIN AREA
    st.markdown("---")
    st.markdown("## 🎯 Asset Selection")
    
    # Categorize assets for better organization
    asset_categories = categorize_assets(asset_columns, descriptions)
    
    # Create columns for asset selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Asset Categories")
        
        # Category selection
        selected_category = st.selectbox(
            "Choose Asset Category",
            list(asset_categories.keys()),
            help="Assets are organized by type for easier selection"
        )
        
        # Asset selection within category
        category_assets = asset_categories[selected_category]
        selected_asset = st.selectbox(
            f"Select Asset from {selected_category}",
            category_assets,
            help=f"{len(category_assets)} assets available in this category"
        )
        
        # Quick stats about the selected asset
        if selected_asset:
            asset_data = filtered_data[selected_asset].dropna()
            if len(asset_data) > 0:
                st.markdown(f"**📈 Data Points:** {len(asset_data)}")
                st.markdown(f"**📅 First Date:** {asset_data.index.min().strftime('%Y-%m-%d')}")
                st.markdown(f"**📅 Last Date:** {asset_data.index.max().strftime('%Y-%m-%d')}")
                try:
                    current_price = float(asset_data.iloc[-1])
                    st.markdown(f"**💰 Current Price:** {current_price:.4f}")
                except (ValueError, TypeError):
                    st.markdown(f"**💰 Current Price:** {asset_data.iloc[-1]}")
    
    with col2:
        st.markdown("### ℹ️ Asset Information")
        
        # Try to find description for selected asset
        asset_description = None
        
        # Get hardcoded descriptions as fallback
        hardcoded_descriptions = get_hardcoded_descriptions()
        
        # Combine loaded descriptions with hardcoded fallback
        all_descriptions = {**hardcoded_descriptions, **descriptions}  # descriptions override hardcoded
        
        # Direct match first
        if selected_asset in all_descriptions and all_descriptions[selected_asset] and not pd.isna(all_descriptions[selected_asset]):
            asset_description = all_descriptions[selected_asset]
        else:
            # Try partial matching for common patterns
            selected_lower = selected_asset.lower()
            for ticker, desc in all_descriptions.items():
                if ticker and desc:
                    # Check if the asset name contains the ticker or vice versa
                    if (selected_lower in ticker.lower() or 
                        ticker.lower() in selected_lower or
                        selected_asset == ticker):
                        asset_description = desc
                        break
        
        if asset_description:
            st.markdown("**Description:**")
            st.info(asset_description)
        else:
            # Show helpful information instead of just "no description"
            st.markdown("**Asset Information:**")
            asset_info = f"Asset: **{selected_asset}**\n\n"
            
            # Try to provide basic info based on asset name patterns (ordered by specificity)
            if any(x in selected_asset.lower() for x in ['h15t', 'usyc', 'ct']):
                asset_info += "🏛️ **Type:** Government bond or treasury security"
            elif 'curncy' in selected_asset.lower():
                asset_info += "📈 **Type:** Currency pair or foreign exchange rate"
            elif 'comdty' in selected_asset.lower():
                asset_info += "🏗️ **Type:** Commodity futures contract"
            elif 'govt' in selected_asset.lower():
                asset_info += "🏛️ **Type:** Government bond or treasury security"
            elif any(x in selected_asset.lower() for x in ['cds', 'cdx', 'itrx']):
                asset_info += "💳 **Type:** Credit Default Swap"
            elif any(x in selected_asset.lower() for x in ['vix', 'vdax', 'move', 'v2x', 'v1x']):
                asset_info += "📈 **Type:** Volatility index"
            elif 'index' in selected_asset.lower():
                asset_info += "📊 **Type:** Market index or equity benchmark"
            else:
                asset_info += "📋 **Type:** Financial instrument"
            
            st.info(asset_info)
        
        # Asset price chart preview
        if selected_asset:
            asset_data = filtered_data[selected_asset].dropna()
            if len(asset_data) > 0:
                st.markdown("**Price Chart Preview:**")
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Scatter(
                    x=asset_data.index, 
                    y=asset_data.values,
                    mode='lines',
                    name=selected_asset,
                    line=dict(color='#1f77b4', width=2)
                ))
                fig_preview.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=f"{selected_asset} Price Evolution",
                    showlegend=False
                )
                st.plotly_chart(fig_preview, use_container_width=True)
    
    st.markdown("---")
    
    # Investment Parameters
    st.sidebar.subheader("💵 Investment Parameters")
    investment_amount = st.sidebar.number_input("Investment Amount per Period", min_value=1, value=1000, step=100)
    frequency = st.sidebar.selectbox("Investment Frequency", ['D', 'W', 'M', 'Y'], 
                                   index=2,  # Default to Monthly (index 2)
                                   format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Y': 'Yearly'}[x])
    
    # Benchmark Selection
    st.sidebar.subheader("📈 Benchmark Configuration")
    benchmark_type = st.sidebar.selectbox("Benchmark Type", ["Fixed Yield", "Other Index"])
    
    benchmark_data = None
    if benchmark_type == "Fixed Yield":
        fixed_yield = st.sidebar.number_input("Annual Yield (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        benchmark_value = fixed_yield
    else:  # Other Index
        benchmark_asset = st.sidebar.selectbox("Select Benchmark Index", asset_columns)
        benchmark_data = filtered_data[benchmark_asset].dropna()
        benchmark_value = None
    
    # Analysis Options
    st.sidebar.subheader("🔍 Analysis Options")
    include_lump_sum = st.sidebar.checkbox("Compare with Lump Sum Investment", value=True)
    calculate_var = st.sidebar.checkbox("Calculate Value at Risk (VaR)", value=True)
    var_confidence = st.sidebar.slider("VaR Confidence Level", 0.01, 0.10, 0.05, 0.01) if calculate_var else 0.05
    
    # Rolling window for relative analysis
    st.sidebar.subheader("📈 Relative Analysis Settings")
    rolling_window = st.sidebar.selectbox("Rolling Window (Months)", [3, 6, 12, 24], index=2, key="rolling_window_sidebar")
    
    # Run Analysis Button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        # Prepare asset data (use filtered data)
        asset_prices = filtered_data[selected_asset].dropna()
        
        # Create benchmark using filtered data
        if benchmark_type == "Fixed Yield":
            benchmark_prices = create_benchmark_series(asset_prices.index, benchmark_type, benchmark_value)
        else:
            # Use filtered benchmark data
            benchmark_data_filtered = filtered_data[benchmark_asset].dropna()
            benchmark_prices = benchmark_data_filtered.reindex(asset_prices.index).fillna(method='ffill')
        
        # Run DCA Simulation
        dca_simulator = DCASimulator(asset_prices, benchmark_prices, investment_amount, frequency)
        dca_results = dca_simulator.simulate()
        dca_summary = dca_simulator.get_performance_summary()
        
        # Run Lump Sum Simulation if requested
        lump_sum_results = None
        lump_sum_summary = None
        if include_lump_sum:
            total_dca_investment = dca_results['total_invested'].iloc[-1]
            lump_sum_simulator = LumpSumSimulator(asset_prices, benchmark_prices, total_dca_investment)
            lump_sum_results = lump_sum_simulator.simulate()
            
            # Calculate lump sum summary
            final_value = lump_sum_results['market_value'].iloc[-1]
            total_invested = lump_sum_results['total_invested'].iloc[0]
            years = (lump_sum_results.index[-1] - lump_sum_results.index[0]).days / 365.25
            
            lump_sum_summary = {
                'Total Invested': total_invested,
                'Market Value': final_value,
                'Total Return (%)': (final_value / total_invested - 1) * 100,
                'Annualized Return (%)': ((final_value / total_invested) ** (1/years) - 1) * 100,
                'Max Drawdown (%)': lump_sum_results['drawdown'].min() * 100
            }
        
        # Display Results
        st.success("✅ Analysis completed successfully!")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance Overview", "📈 Detailed Charts", "🔥 Relative Analysis", "📋 Risk Analysis", "📑 Data Export"])
        
        with tab1:
            # Performance Overview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 DCA Performance")
                for metric, value in dca_summary.items():
                    if isinstance(value, (int, float)):
                        if 'Return' in metric or 'IRR' in metric or 'Drawdown' in metric or 'Volatility' in metric:
                            st.metric(metric, f"{value:.2f}%")
                        elif 'Ratio' in metric:
                            st.metric(metric, f"{value:.2f}")
                        else:
                            st.metric(metric, f"${value:,.2f}")
            
            with col2:
                if include_lump_sum and lump_sum_summary:
                    st.subheader("💰 Lump Sum Performance")
                    for metric, value in lump_sum_summary.items():
                        if isinstance(value, (int, float)):
                            if 'Return' in metric or 'Drawdown' in metric:
                                st.metric(metric, f"{value:.2f}%")
                            else:
                                st.metric(metric, f"${value:,.2f}")
            
            # Performance Comparison Chart
            fig_overview = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Value Over Time', 'Performance Index Comparison', 
                              'Drawdown Analysis', 'Monthly Returns Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Portfolio Value
            fig_overview.add_trace(
                go.Scatter(x=dca_results.index, y=dca_results['total_invested'], 
                          name='Total Invested', line=dict(color='blue')),
                row=1, col=1
            )
            fig_overview.add_trace(
                go.Scatter(x=dca_results.index, y=dca_results['market_value'], 
                          name='DCA Market Value', line=dict(color='green')),
                row=1, col=1
            )
            
            if include_lump_sum and lump_sum_results is not None:
                fig_overview.add_trace(
                    go.Scatter(x=lump_sum_results.index, y=lump_sum_results['market_value'], 
                              name='Lump Sum Value', line=dict(color='orange')),
                    row=1, col=1
                )
            
            # Performance Index
            fig_overview.add_trace(
                go.Scatter(x=dca_results.index, y=dca_results['portfolio_value_index'], 
                          name='DCA Index', line=dict(color='green')),
                row=1, col=2
            )
            fig_overview.add_trace(
                go.Scatter(x=dca_results.index, y=dca_results['benchmark_index'], 
                          name='Benchmark', line=dict(color='red')),
                row=1, col=2
            )
            
            if include_lump_sum and lump_sum_results is not None:
                fig_overview.add_trace(
                    go.Scatter(x=lump_sum_results.index, y=lump_sum_results['portfolio_value_index'], 
                              name='Lump Sum Index', line=dict(color='orange')),
                    row=1, col=2
                )
            
            # Drawdown
            fig_overview.add_trace(
                go.Scatter(x=dca_results.index, y=dca_results['drawdown'] * 100, 
                          name='DCA Drawdown', fill='tonexty', line=dict(color='red')),
                row=2, col=1
            )
            
            # Returns Distribution
            returns = dca_simulator.asset_returns * 100
            fig_overview.add_trace(
                go.Histogram(x=returns, name='Monthly Returns', nbinsx=30),
                row=2, col=2
            )
            
            fig_overview.update_layout(height=800, showlegend=True, title_text="Investment Analysis Overview")
            st.plotly_chart(fig_overview, use_container_width=True)
        
        with tab2:
            # Detailed Charts
            st.subheader("📈 Detailed Performance Analysis")
            
            # Asset Price Chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=asset_prices.index, y=asset_prices.values, 
                                         name=f'{selected_asset} Price', line=dict(color='blue')))
            fig_price.update_layout(title=f'{selected_asset} Price History', 
                                  xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Cumulative Investment vs Value
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(x=dca_results.index, y=dca_results['total_invested'], 
                                              name='Cumulative Investment', line=dict(color='blue')))
            fig_cumulative.add_trace(go.Scatter(x=dca_results.index, y=dca_results['market_value'], 
                                              name='Portfolio Value', line=dict(color='green')))
            
            # Add profit/loss area
            fig_cumulative.add_trace(go.Scatter(
                x=dca_results.index.tolist() + dca_results.index.tolist()[::-1],
                y=dca_results['total_invested'].tolist() + dca_results['market_value'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,255,0,0.2)' if dca_results['market_value'].iloc[-1] > dca_results['total_invested'].iloc[-1] else 'rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Profit/Loss Area',
                showlegend=False
            ))
            
            fig_cumulative.update_layout(title='Cumulative Investment vs Portfolio Value', 
                                       xaxis_title='Date', yaxis_title='Value ($)')
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with tab3:
            # Relative Analysis
            st.subheader("🔥 Monthly Relative Performance Analysis")
            st.markdown("**Green indicates DCA outperformed, Red indicates DCA underperformed**")
            
            # Create and display heatmaps
            heatmap_fig = create_relative_performance_heatmaps(dca_simulator, lump_sum_results)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Rolling relative performance
            st.subheader("📈 Rolling Relative Performance Trends")
            st.markdown(f"**Current Rolling Window: {rolling_window} months** (Change in sidebar)")
            
            rolling_fig = create_rolling_relative_performance_chart(dca_simulator, lump_sum_results, rolling_window)
            st.plotly_chart(rolling_fig, use_container_width=True)
            
            # Annual performance summary
            st.subheader("📋 Annual Relative Performance Summary")
            annual_summary = create_annual_relative_performance_summary(dca_simulator, lump_sum_results)
            
            # Format column names for better display
            display_columns = {
                'Rel_Benchmark_Mean': 'Avg Rel vs Benchmark (%)',
                'Rel_Benchmark_Std': 'Volatility vs Benchmark (%)',
                'Rel_Benchmark_Min': 'Min Rel vs Benchmark (%)',
                'Rel_Benchmark_Max': 'Max Rel vs Benchmark (%)',
                'DCA_Return_Mean': 'Avg DCA Return (%)',
                'DCA_Return_Std': 'DCA Volatility (%)',
                'Benchmark_Return_Mean': 'Avg Benchmark Return (%)',
                'Benchmark_Return_Std': 'Benchmark Volatility (%)'
            }
            
            if include_lump_sum and lump_sum_results is not None:
                display_columns.update({
                    'Rel_LumpSum_Mean': 'Avg Rel vs Lump Sum (%)',
                    'Rel_LumpSum_Std': 'Volatility vs Lump Sum (%)',
                    'Rel_LumpSum_Min': 'Min Rel vs Lump Sum (%)',
                    'Rel_LumpSum_Max': 'Max Rel vs Lump Sum (%)'
                })
            
            # Select and rename columns for display
            display_df = annual_summary[[col for col in display_columns.keys() if col in annual_summary.columns]].copy()
            display_df.columns = [display_columns[col] for col in display_df.columns]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Performance insights
            st.subheader("💡 Key Insights")
            
            # Calculate overall statistics
            rel_perf_data = dca_simulator.calculate_relative_performance(lump_sum_results)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_rel_benchmark = rel_perf_data['relative_vs_benchmark'].mean() * 100
                positive_months_benchmark = (rel_perf_data['relative_vs_benchmark'] > 0).sum()
                total_months = len(rel_perf_data['relative_vs_benchmark'].dropna())
                
                st.metric("Avg Monthly Outperformance vs Benchmark", f"{avg_rel_benchmark:.2f}%")
                st.metric("Months Outperforming Benchmark", f"{positive_months_benchmark}/{total_months}")
            
            with col2:
                if include_lump_sum and 'relative_vs_lumpsum' in rel_perf_data.columns:
                    avg_rel_lumpsum = rel_perf_data['relative_vs_lumpsum'].mean() * 100
                    positive_months_lumpsum = (rel_perf_data['relative_vs_lumpsum'] > 0).sum()
                    
                    st.metric("Avg Monthly Outperformance vs Lump Sum", f"{avg_rel_lumpsum:.2f}%")
                    st.metric("Months Outperforming Lump Sum", f"{positive_months_lumpsum}/{total_months}")
            
            with col3:
                best_year = annual_summary['Rel_Benchmark_Mean'].idxmax()
                worst_year = annual_summary['Rel_Benchmark_Mean'].idxmin()
                
                st.metric("Best Relative Year vs Benchmark", f"{best_year}")
                st.metric(f"Performance in {best_year}", f"{annual_summary.loc[best_year, 'Rel_Benchmark_Mean']:.2f}%")
            
            # Distribution Analysis
            st.subheader("📊 Performance Distribution Analysis")
            distribution_fig = create_performance_distribution_chart(dca_simulator, lump_sum_results)
            st.plotly_chart(distribution_fig, use_container_width=True)
            
            # Win/Loss Analysis
            st.subheader("🎯 Win/Loss Analysis")
            win_loss_df = create_win_loss_analysis(dca_simulator, lump_sum_results)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(win_loss_df, use_container_width=True)
            
            with col2:
                # Create win rate visualization
                if len(win_loss_df) > 0:
                    fig_winrate = go.Figure(data=[
                        go.Bar(
                            x=win_loss_df['Strategy'],
                            y=win_loss_df['Win Rate (%)'],
                            text=[f"{rate:.1f}%" for rate in win_loss_df['Win Rate (%)']],
                            textposition='auto',
                            marker_color=['blue', 'orange'] if len(win_loss_df) > 1 else ['blue']
                        )
                    ])
                    
                    fig_winrate.update_layout(
                        title='Win Rate Comparison',
                        yaxis_title='Win Rate (%)',
                        height=400,
                        showlegend=False
                    )
                    fig_winrate.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.7)
                    st.plotly_chart(fig_winrate, use_container_width=True)
        
        with tab4:
            # Risk Analysis
            st.subheader("⚠️ Risk Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            returns = dca_simulator.asset_returns
            
            with col1:
                if calculate_var:
                    daily_var = RiskMetrics.calculate_var(returns, var_confidence) * 100
                    st.metric(f"Daily VaR ({var_confidence*100:.0f}%)", f"{daily_var:.2f}%")
                    
                    daily_cvar = RiskMetrics.calculate_cvar(returns, var_confidence) * 100
                    st.metric(f"Daily CVaR ({var_confidence*100:.0f}%)", f"{daily_cvar:.2f}%")
            
            with col2:
                sortino_ratio = RiskMetrics.calculate_sortino_ratio(returns)
                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                
                skewness = returns.skew()
                st.metric("Return Skewness", f"{skewness:.2f}")
            
            with col3:
                kurtosis = returns.kurtosis()
                st.metric("Return Kurtosis", f"{kurtosis:.2f}")
                
                positive_months = (returns > 0).sum() / len(returns) * 100
                st.metric("Positive Return Periods", f"{positive_months:.1f}%")
            
            # Risk-Return Scatter
            if include_lump_sum:
                strategies = ['DCA', 'Lump Sum']
                returns_data = [dca_summary['Annualized Return (%)'], lump_sum_summary['Annualized Return (%)']]
                volatility_data = [dca_summary['Annualized Volatility (%)'], 
                                 returns.std() * np.sqrt(12) * 100]  # Use same volatility calculation for lump sum
                
                fig_risk_return = go.Figure(data=go.Scatter(
                    x=volatility_data,
                    y=returns_data,
                    mode='markers+text',
                    text=strategies,
                    textposition="top center",
                    marker=dict(size=20, color=['green', 'orange'])
                ))
                
                fig_risk_return.update_layout(
                    title='Risk-Return Analysis',
                    xaxis_title='Annualized Volatility (%)',
                    yaxis_title='Annualized Return (%)'
                )
                st.plotly_chart(fig_risk_return, use_container_width=True)
        
        with tab5:
            # Data Export
            st.subheader("📑 Export Results")
            
            # Export format selection
            export_format = st.selectbox("Export Format", ["CSV (Excel compatible)", "CSV (same as input)", "Excel"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download DCA Results"):
                    if export_format == "Excel":
                        # For Excel export, we need to use BytesIO
                        from io import BytesIO
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            dca_results.to_excel(writer, sheet_name='DCA_Results')
                        
                        st.download_button(
                            label="Download DCA Excel",
                            data=buffer.getvalue(),
                            file_name=f"dca_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    elif export_format == "CSV (same as input)" and file_type == "CSV":
                        # Use same format as input CSV
                        csv = dca_results.to_csv(sep=csv_params['sep'], decimal=csv_params['decimal'])
                        st.download_button(
                            label="Download DCA CSV",
                            data=csv,
                            file_name=f"dca_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        # Default CSV (Excel compatible)
                        csv = dca_results.to_csv(sep=',', decimal='.')
                        st.download_button(
                            label="Download DCA CSV",
                            data=csv,
                            file_name=f"dca_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
            
            with col2:
                if include_lump_sum and lump_sum_results is not None:
                    if st.button("Download Lump Sum Results"):
                        if export_format == "Excel":
                            from io import BytesIO
                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                lump_sum_results.to_excel(writer, sheet_name='LumpSum_Results')
                            
                            st.download_button(
                                label="Download Lump Sum Excel",
                                data=buffer.getvalue(),
                                file_name=f"lump_sum_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif export_format == "CSV (same as input)" and file_type == "CSV":
                            csv = lump_sum_results.to_csv(sep=csv_params['sep'], decimal=csv_params['decimal'])
                            st.download_button(
                                label="Download Lump Sum CSV",
                                data=csv,
                                file_name=f"lump_sum_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            csv = lump_sum_results.to_csv(sep=',', decimal='.')
                            st.download_button(
                                label="Download Lump Sum CSV",
                                data=csv,
                                file_name=f"lump_sum_results_{selected_asset}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
            
            # Summary Statistics Table
            st.subheader("📊 Summary Statistics")
            summary_df = pd.DataFrame([dca_summary])
            if include_lump_sum and lump_sum_summary:
                lump_sum_df = pd.DataFrame([lump_sum_summary])
                summary_df = pd.concat([summary_df, lump_sum_df], keys=['DCA', 'Lump Sum'])
            
            st.dataframe(summary_df, use_container_width=True)

if __name__ == "__main__":
    main()