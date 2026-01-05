import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import os
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="NSE UDiFF Dashboard", layout="wide")
CACHE_DIR = "data_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- CACHED DOWNLOADER ---
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

# --- MAIN UI ---
st.title("📊 NSE Historical Volatility Hub (UDiFF)")

with st.sidebar:
    st.header("Settings")
    picked_date = st.date_input("Select Date", value=datetime(2025, 12, 31))
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])

data = get_nifty_data(picked_date)

if isinstance(data, pd.DataFrame):
    # Filter for specific symbol and options
    # UDiFF Headers: TckrSymb (Symbol), OptnTp (Type), StrkPric (Strike)
    df_filtered = data[data['TckrSymb'] == symbol].copy()
    
    st.success(f"Loaded {len(df_filtered)} contracts for {symbol}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Open Interest", f"{df_filtered['OpnIntrst'].sum():,.0f}")
    with col2:
        st.metric("Unique Strikes", len(df_filtered['StrkPric'].unique()))

    st.dataframe(df_filtered, use_container_width=True)
else:
    st.error(data)
