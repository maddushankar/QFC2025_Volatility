import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import zipfile
import io
import os
from datetime import datetime
from scipy.optimize import brentq
import numpy as np
from scipy.stats import norm

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Maturity Hub", layout="wide")
CACHE_DIR = "data_cache"
if not os.path.exists(CACHE_DIR) : os.makedirs(CACHE_DIR)
    
# --- INITIALIZE SESSION STATE ---
# This prevents the AttributeError
if 'active_df' not in st.session_state: st.session_state.active_df = None


# --- BLACK-SCHOLES ENGINE ---
def black_scholes(S, K, T, r, sigma, option_type='CE'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def find_iv(market_price, S, K, T, r, option_type):
    # Rule 1: Ignore junk prices
    if market_price <= 0.5: 
        return 0.0
    
    # Rule 2: Define the target function (Raw difference, not squared)
    def objective_function(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    #try:
        # Use brentq - it is much more stable than newton for IV
        # It looks for a solution between 1% and 500% volatility
    return brentq(objective_function, 0.01, 5.0, xtol=1e-5)
    #except (ValueError, RuntimeError):
        # If the price is mathematically impossible (e.g. below intrinsic value)
        #return 0.0
def get_tte(trade_date_str, expiry_date_str):
    t = datetime.strptime(trade_date_str, '%Y-%m-%d')
    e = datetime.strptime(expiry_date_str, '%Y-%m-%d')
    days = (e - t).days
    return max(days, 1) / 365.0  # Time in years
    
# --- CORE FUNCTION: Download & Extract ---
def get_nifty_data(target_date):
    date_str = target_date.strftime("%Y%m%d")
    # Official NSE UDiFF URL Pattern
    url = f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_str}_F_0000.csv.zip"
    local_filename = f"{CACHE_DIR}/fo_{date_str}.csv"

    # Step 1: Check Disk Cache First
    if os.path.exists(local_filename):
        return pd.read_csv(local_filename)

    # Step 2: Download if not in Cache
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.nseindia.com/"
    }
    
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5) # Handshake
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f)
                    df.columns = [c.strip() for c in df.columns]
                    # Map new ISO Tags to readable names immediately
                    cols = {
                        'TckrSymb': 'SYMBOL', 'XpryDt': 'EXPIRY', 'StrkPric': 'STRIKE',
                        'OptnTp': 'TYPE', 'ClsPric': 'CLOSE', 'OpnIntrst': 'OI', 'UndrlygPric': 'SPOT'
                    }
                    df = df.rename(columns=cols)
                    df.to_csv(local_filename, index=False)
                    return df
        else:
            return f"NSE Error: Status {response.status_code}. Is it a holiday?"
    except Exception as e:
        return f"Request failed: {str(e)}"

# --- SIDEBAR & SESSION STATE ---
if 'active_df' not in st.session_state:
    st.session_state.active_df = None

with st.sidebar:
    st.header("📅 Data Controls")
    trading_date = st.date_input("Trading Day", value=datetime(2025, 12, 31))
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
    
    # TRIGGER BUTTON
    if st.button("🚀 Get Data", use_container_width=True):
        with st.spinner("Fetching from NSE Archives..."):
            result = get_nifty_data(trading_date)
            if isinstance(result, pd.DataFrame):
                st.session_state.active_df = result
                st.success("Data Loaded!")
            else:
                st.error(result)

    # 2. Risk-Free Rate Input
    # We use format="%.2f" to show two decimal places
    risk_free_percent = st.number_input(
        "Risk-Free Rate (%)", 
        min_value=0.0, 
        max_value=15.0, 
        value=7.0, 
        step=0.05,
        format="%.2f",
        help="The annual yield of a risk-free bond (like a Govt Treasury Bond)."
    )
    
    # Convert the percentage to a decimal for the Black-Scholes math
    risk_free = risk_free_percent / 100
    
    st.divider()

# --- MAIN DISPLAY LOGIC ---
if st.session_state.active_df is not None:
    df = st.session_state.active_df
    
    # Filter for Symbol
    df_symbol = df[df['SYMBOL'] == symbol].copy()
    
    # 1. Maturity Selector (Now that data exists)
    all_expiries = sorted(df_symbol['EXPIRY'].unique())
    selected_expiry = st.sidebar.selectbox("Select Maturity", all_expiries)
    
    # 2. Final Filtered Data
    final_df = df_symbol[df_symbol['EXPIRY'] == selected_expiry].sort_values("STRIKE")

    # Calculate IV for each row
    with st.spinner("Calculating Implied Volatility..."):
        
        tte = get_tte(str(trading_date), selected_expiry)
        final_df['intrinsic'] = np.where(
            final_df['TYPE'] == 'CE',
            (final_df['SPOT'] - final_df['STRIKE'] * np.exp(-risk_free * tte)), # Discounted Strike
            (final_df['STRIKE'] * np.exp(-risk_free * tte) - final_df['SPOT']))
        final_df['intrinsic'] = final_df['intrinsic'].clip(lower=0)
        final_df = final_df[final_df['CLOSE'] > final_df['intrinsic']].copy()
        final_df = final_df[final_df['CLOSE'] > 1.0]
        final_df['IV'] = final_df.apply(
            lambda row: find_iv(row['CLOSE'], row['SPOT'], row['STRIKE'], tte, risk_free, row['TYPE']), axis=1
        )
        # Convert to percentage
        final_df['IV_pct'] = final_df['IV'] * 100
        
    # 3. Visualization
    st.subheader(f"📈 {symbol} Close Prices - Expiry: {selected_expiry}")
    
    fig = px.line(final_df, x="STRIKE", y="CLOSE", color="TYPE",
                  labels={"STRIKE": "Strike Price", "CLOSE": "Closing Price"},
                  markers=True, template="plotly_dark")
    fig = px.line(final_df, x="STRIKE", y="IV_pct", color="TYPE",
                  labels={"STRIKE": "Strike Price", "IV_Pct": "Implied Volatility (BS)"},
                  markers=True, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 4. Data Table
    with st.expander("View Filtered Data Table"):
        st.dataframe(final_df[['STRIKE', 'TYPE', 'CLOSE', 'IV_pct', 'OI']], use_container_width=True)
else:
    st.info("Please select a date and click 'Get Data' in the sidebar to begin.")
