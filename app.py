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
        default_path = r"X:\Stagiaires\Vincent N\data\data\risk_proxies\all_risk_proxies.csv"
        description_path = r"X:\Stagiaires\Vincent N\data\data\risk_proxies\data_description.csv"
        
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
                descriptions = dict(zip(desc_df['Ticker'], desc_df['Description']))
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
    
    elif benchmark_type == "Money Market Fund" and benchmark_data is not None:
        return benchmark_data
    
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

def categorize_assets(asset_columns, descriptions):
    """Categorize assets by type for better organization"""
    categories = {
        "Equities & Indices": [],
        "Currencies": [],
        "Commodities": [],
        "Bonds & Rates": [],
        "Credit (CDS)": [],
        "Volatility": [],
        "Other": []
    }
    
    for asset in asset_columns:
        asset_lower = asset.lower()
        desc = descriptions.get(asset, "").lower()
        
        if any(x in asset_lower for x in ['index', 'spx', 'eses', 'deda', 'asda', 'vg1', 'wlsnre', 'spw']):
            categories["Equities & Indices"].append(asset)
        elif any(x in asset_lower for x in ['curncy', 'eur', 'usd', 'jpy', 'chf', 'gbp', 'aud', 'nzd']):
            categories["Currencies"].append(asset)
        elif any(x in asset_lower for x in ['comdty', 'xau', 'xpt', 'si1', 'era', 'oata', 'cla', 'hg']):
            categories["Commodities"].append(asset)
        elif any(x in asset_lower for x in ['govt', 'h15t', 'usyc', 'ct', 'gt', 'guk']):
            categories["Bonds & Rates"].append(asset)
        elif any(x in asset_lower for x in ['cds', 'cdx', 'itrx']):
            categories["Credit (CDS)"].append(asset)
        elif any(x in asset_lower for x in ['vix', 'vdax', 'v2x', 'v1x', 'move']):
            categories["Volatility"].append(asset)
        else:
            categories["Other"].append(asset)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 DCA Investment Simulator</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Dollar Cost Averaging Analysis Tool")
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sidebar-header">📊 Configuration</h2>', unsafe_allow_html=True)
    
    # Data Source Selection
    st.sidebar.subheader("📁 Data Source")
    data_source = st.sidebar.selectbox("Data Source", ["Default Risk Proxies", "Upload Custom File"])
    
    data = None
    descriptions = {}
    
    if data_source == "Default Risk Proxies":
        # Try to load default data
        data, descriptions = load_default_data()
        if data is None:
            st.error("⚠️ Could not load default risk proxies data. Please check the file path or upload a custom file.")
            st.sidebar.markdown("**Fallback to file upload:**")
            data_source = "Upload Custom File"
    
    if data_source == "Upload Custom File" or data is None:
        # File type selection
        file_type = st.sidebar.selectbox("File Type", ["Excel", "CSV"])
        
        if file_type == "Excel":
            uploaded_file = st.sidebar.file_uploader("Upload Excel file with asset data", type=['xlsx', 'xls'])
        else:
            uploaded_file = st.sidebar.file_uploader("Upload CSV file with asset data", type=['csv'])
        
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
        
        # Direct match first
        if selected_asset in descriptions and descriptions[selected_asset]:
            asset_description = descriptions[selected_asset]
        else:
            # Try partial matching for common patterns
            selected_lower = selected_asset.lower()
            for ticker, desc in descriptions.items():
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
            
            # Try to provide basic info based on asset name patterns
            if 'curncy' in selected_asset.lower():
                asset_info += "📈 **Type:** Currency pair or foreign exchange rate"
            elif 'index' in selected_asset.lower():
                asset_info += "📊 **Type:** Market index or equity benchmark"
            elif 'comdty' in selected_asset.lower():
                asset_info += "🏗️ **Type:** Commodity futures contract"
            elif 'govt' in selected_asset.lower():
                asset_info += "🏛️ **Type:** Government bond or treasury security"
            elif any(x in selected_asset.lower() for x in ['cds', 'cdx', 'itrx']):
                asset_info += "💳 **Type:** Credit Default Swap"
            elif any(x in selected_asset.lower() for x in ['vix', 'vdax', 'move']):
                asset_info += "📈 **Type:** Volatility index"
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
                                   format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Y': 'Yearly'}[x])
    
    # Benchmark Selection
    st.sidebar.subheader("📈 Benchmark Configuration")
    benchmark_type = st.sidebar.selectbox("Benchmark Type", ["Fixed Yield", "Money Market Fund", "Other Index"])
    
    benchmark_data = None
    if benchmark_type == "Fixed Yield":
        fixed_yield = st.sidebar.number_input("Annual Yield (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
        benchmark_value = fixed_yield
    elif benchmark_type == "Money Market Fund":
        benchmark_asset = st.sidebar.selectbox("Select Money Market Fund", asset_columns)
        benchmark_data = filtered_data[benchmark_asset].dropna()
        benchmark_value = None
    else:  # Other Index
        benchmark_asset = st.sidebar.selectbox("Select Benchmark Index", asset_columns)
        benchmark_data = filtered_data[benchmark_asset].dropna()
        benchmark_value = None
    
    # Analysis Options
    st.sidebar.subheader("🔍 Analysis Options")
    include_lump_sum = st.sidebar.checkbox("Compare with Lump Sum Investment", value=True)
    calculate_var = st.sidebar.checkbox("Calculate Value at Risk (VaR)", value=True)
    var_confidence = st.sidebar.slider("VaR Confidence Level", 0.01, 0.10, 0.05, 0.01) if calculate_var else 0.05
    
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Overview", "📈 Detailed Charts", "📋 Risk Analysis", "📑 Data Export"])
        
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
        
        with tab4:
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