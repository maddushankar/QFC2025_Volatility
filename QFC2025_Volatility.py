import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Maturity Hub", layout="wide")
CACHE_DIR = "data_cache"

# --- SIDEBAR: Controls ---
with st.sidebar:
    st.header("📅 Data Controls")
    # Step 1: Select the Trading Day (The day the data was recorded)
    trading_date = st.date_input("Trading Day", value=datetime(2025, 12, 31))
    symbol = st.selectbox("Index", ["NIFTY", "BANKNIFTY"])
    
    st.divider()
    st.info("Download the day's file first to see available maturities.")

# --- CORE FUNCTION: Load Data ---
@st.cache_data
def load_and_standardize(date_obj):
    date_str = date_obj.strftime("%Y%m%d")
    file_path = f"{CACHE_DIR}/fo_{date_str}.csv"
    
    if not os.path.exists(file_path):
        return None # In a real app, call your download function here
        
    df = pd.read_csv(file_path)
    # Mapping UDiFF ISO Tags to readable names
    cols = {
        'TckrSymb': 'SYMBOL',
        'XpryDt': 'EXPIRY',
        'StrkPric': 'STRIKE',
        'OptnTp': 'TYPE',
        'ClsPric': 'CLOSE',
        'OpnIntrst': 'OI'
    }
    df = df.rename(columns=cols)
    return df

# --- MAIN LOGIC ---
df = load_and_standardize(trading_date)

if df is not None:
    # Filter for NIFTY/BANKNIFTY
    df_symbol = df[df['SYMBOL'] == symbol].copy()
    
    # Step 2: Dynamically update sidebar with available Expiries
    available_expiries = sorted(df_symbol['EXPIRY'].unique())
    selected_expiry = st.sidebar.selectbox("Maturity / Expiry", available_expiries)
    
    # Final Filtering
    final_df = df_symbol[df_symbol['EXPIRY'] == selected_expiry].sort_values("STRIKE")
    
    # --- VISUALIZATION ---
    st.subheader(f"Close Prices for {symbol} - Expiry: {selected_expiry}")
    
    # Comparison Chart: CE vs PE Close Prices
    fig = px.line(final_df, x="STRIKE", y="CLOSE", color="TYPE",
                  title=f"Closing Prices by Strike ({selected_expiry})",
                  labels={"STRIKE": "Strike Price", "CLOSE": "Closing Price (LTP)"},
                  markers=True)
    
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table
    with st.expander("View Raw Data Table"):
        st.dataframe(final_df[['STRIKE', 'TYPE', 'CLOSE', 'OI']], use_container_width=True)
else:
    st.warning(f"No cached data found for {trading_date}. Please trigger a download first.")
