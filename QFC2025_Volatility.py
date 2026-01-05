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
def black_scholes(F, K, T, sigma, option_type='CE'):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return (F * norm.cdf(d1) - K * norm.cdf(d2)) * np.exp(-0.05 * T) # Disount factor is negligible for IV shape
    else:
        return (K * norm.cdf(-d2) - F * norm.cdf(-d1)) * np.exp(-0.05 * T)

def find_iv(market_price, F, K, T, option_type):
    if market_price <= 0.5: return 0.0
    
    def objective(sigma):
        return black_scholes(F, K, T, sigma, option_type) - market_price

    try:
        # Solving for sigma between 0.1% and 500%
        return brentq(objective, 0.001, 5.0, xtol=1e-5)
    except:
        return np.nan
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

    # # 2. Risk-Free Rate Input
    # # We use format="%.2f" to show two decimal places
    # risk_free_percent = st.number_input(
    #     "Risk-Free Rate (%)", 
    #     min_value=0.0, 
    #     max_value=15.0, 
    #     value=7.0, 
    #     step=0.05,
    #     format="%.2f",
    #     help="The annual yield of a risk-free bond (like a Govt Treasury Bond)."
    # )
    
    # # Convert the percentage to a decimal for the Black-Scholes math
    # risk_free = risk_free_percent / 100
    
    min_volume = st.number_input(
        "Minimum Volume", 
        min_value=0, 
        max_value=10000000, 
        value=10000 ,
        step=1000,
        format="%d",
        help="Minimum volume that the option contracts to have to be considered in the analysis."
    )
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
        strike_range = 0.2
        # min_volume = 10000
        spot_price = final_df['SPOT'].iloc[0]
        upper_bound = spot_price * (1 + strike_range)
        lower_bound = spot_price * (1 - strike_range)
        
        final_df = final_df[
            (final_df['STRIKE'] >= lower_bound) & 
            (final_df['STRIKE'] <= upper_bound)
        ].copy()
        final_df = final_df[final_df['OI'] > min_volume]
        # 1. Identify the ATM Strike
        
        atm_strike = final_df.iloc[(final_df['STRIKE'] - spot_price).abs().argsort()[:1]]['STRIKE'].values[0]
        
        # 2. Filter using the ATM strike as the pivot
        otm_puts = final_df[(final_df['TYPE'] == 'PE') & (final_df['STRIKE'] <= atm_strike)]
        otm_calls = final_df[(final_df['TYPE'] == 'CE') & (final_df['STRIKE'] >= atm_strike)]
        atm_call = final_df[(final_df['STRIKE'] == atm_strike) & (final_df['TYPE'] == 'CE')]['CLOSE'].mean()
        atm_put = final_df[(final_df['STRIKE'] == atm_strike) & (final_df['TYPE'] == 'PE')]['CLOSE'].mean()
        synthetic_fwd = atm_strike + (atm_call - atm_put)
        
        # 3. Combine
        smile_df = pd.concat([otm_puts, otm_calls]).drop_duplicates(subset=['STRIKE', 'TYPE'])

        # final_df = pd.concat([final_df[(final_df['TYPE']=='PE')&(final_df['STRIKE']<final_df['SPOT'])],\
        #                       final_df[(final_df['TYPE']=='CE')&(final_df['STRIKE']>final_df['SPOT'])]])
        # smile_df['intrinsic'] = np.where(
        #     smile_df['TYPE'] == 'CE',
        #     (smile_df['SPOT'] - smile_df['STRIKE'] * np.exp(-risk_free * tte)), # Discounted Strike
        #     (smile_df['STRIKE'] * np.exp(-risk_free * tte) - smile_df['SPOT']))
        smile_df['intrinsic'] = np.where(smile_df['TYPE'] == 'CE', 
                                     (synthetic_fwd - smile_df['STRIKE']), 
                                     (smile_df['STRIKE'] - synthetic_fwd))
        
        smile_df['intrinsic'] = smile_df['intrinsic'].clip(lower=0)
        smile_df = smile_df[smile_df['CLOSE'] > smile_df['intrinsic']].copy()
        smile_df = smile_df[smile_df['CLOSE'] > 1.0]
        smile_df['IV'] = smile_df.apply(
            lambda row: find_iv(row['CLOSE'], synthetic_fwd, row['STRIKE'], tte, row['TYPE']), axis=1
        )
        # Convert to percentage
        smile_df['IV_pct'] = smile_df['IV'] * 100
        
    # 3. Visualization
#    st.subheader(f"📈 {symbol} Close Prices - Expiry: {selected_expiry}")
#    
#    fig = px.line(final_df, x="STRIKE", y="CLOSE", color="TYPE",
#                  labels={"STRIKE": "Strike Price", "CLOSE": "Closing Price"},
#                  markers=True, template="plotly_dark")
#    fig = px.line(final_df, x="STRIKE", y="IV_pct", color="TYPE",
#                  labels={"STRIKE": "Strike Price", "IV_Pct": "Implied Volatility (BS)"},
#                  markers=True, template="plotly_dark")
#    st.plotly_chart(fig, use_container_width=True)
# --- 3. SIDE-BY-SIDE VISUALIZATION ---
    st.subheader(f"📊 {symbol} Analysis - Expiry: {selected_expiry}")

    # Create two equal-width columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot 1: Close Prices
        fig_price = px.line(
            final_df, 
            x="STRIKE", 
            y="CLOSE", 
            color="TYPE",
            title="Option Closing Prices",
            labels={"STRIKE": "Strike", "CLOSE": "Price (₹)"},
            color_discrete_map={'CE': '#00ff00', 'PE': '#ff0000'},
            markers=True, 
            template="plotly_dark"
        )
        fig_price.update_layout(hovermode="x unified")
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Plot 2: Implied Volatility (The Smile)
        fig_iv = px.line(
            smile_df, 
            x="STRIKE", 
            y="IV_pct", 
            
            title="Implied Volatility (Smile) - OTM CE and OTM PE",
            labels={"STRIKE": "Strike", "IV_pct": "IV (%)"},
            markers=False, 
            template="plotly_dark"
        )
        fig_iv.update_layout(hovermode="x unified")
        
        # 3. Add the Spot Line and Layout updates
        current_spot = smile_df['SPOT'].iloc[0]
        fig_iv.add_vline(x=current_spot, line_dash="dash", line_color="grey", annotation_text="SPOT")
        fig_iv.update_layout(hovermode="x unified")
        st.plotly_chart(fig_iv, use_container_width=True)
        # 4. Data Table
    with st.expander("View Filtered Data Table"):
        st.dataframe(smile_df[['STRIKE', 'TYPE', 'CLOSE', 'IV_pct', 'OI']].sort_values(by=['TYPE','STRIKE']), use_container_width=True)
else:
    st.info("Please select a date and click 'Get Data' in the sidebar to begin.")
