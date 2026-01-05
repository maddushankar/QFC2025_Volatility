import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Nifty Maturity Hub", layout="wide")
CACHE_DIR = "data_cache"

@st.cache_data(show_spinner="Downloading from NSE...")
def get_nifty_data(target_date):
    date_str = target_date.strftime("%Y%m%d")
    # New UDiFF URL Pattern
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
                    # Standardize UDiFF columns for easier use
                    df.columns = [c.strip() for c in df.columns]
                    # Save to Disk Cache for future reboots
                    df.to_csv(local_filename, index=False)
                    return df
        else:
            return f"Error: Status {response.status_code} (Market likely closed)"
    except Exception as e:
        return f"Request failed: {str(e)}"
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
    data = get_nifty_data(date_obj)
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
