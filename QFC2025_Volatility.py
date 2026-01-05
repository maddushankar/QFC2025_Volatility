import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(page_title="Team Volatility Hub", layout="wide")

# --- CUSTOM BROWSER SESSION ---
# This prevents the 'YFRateLimitError' by mimicking a browser
@st.cache_resource
def get_browser_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

# --- DATA FETCHING: NSE (Manual Session) ---
@st.cache_data(ttl=60)
def get_nse_option_chain(symbol):
    session = get_browser_session()
    base_url = "https://www.nseindia.com/option-chain"
    # Handshake to get cookies
    session.get(base_url, timeout=10)
    
    api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    if symbol not in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        api_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    
    try:
        response = session.get(api_url, timeout=10)
        data = response.json().get('filtered', {}).get('data', [])
        rows = []
        for r in data:
            rows.append({
                "Strike": r['strikePrice'],
                "CE_OI": r.get('CE', {}).get('openInterest', 0),
                "CE_LTP": r.get('CE', {}).get('lastPrice', 0),
                "PE_LTP": r.get('PE', {}).get('lastPrice', 0),
                "PE_OI": r.get('PE', {}).get('openInterest', 0)
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

# --- DATA FETCHING: SPX (yfinance with Session) ---
@st.cache_data(ttl=300)
def get_spx_option_chain(expiry_date):
    session = get_browser_session()
    ticker = yf.Ticker("^SPX", session=session)
    try:
        chain = ticker.option_chain(expiry_date)
        calls = chain.calls[['strike', 'lastPrice', 'openInterest']].rename(columns={'lastPrice': 'CE_LTP', 'openInterest': 'CE_OI'})
        puts = chain.puts[['strike', 'lastPrice', 'openInterest']].rename(columns={'lastPrice': 'PE_LTP', 'openInterest': 'PE_OI'})
        df = pd.merge(calls, puts, on='strike', how='inner')
        return df.rename(columns={'strike': 'Strike'})
    except:
        return pd.DataFrame()

# --- MAIN UI ---
st.title("📈 Global Volatility Dashboard")
market_choice = st.sidebar.radio("Select Market", ["India (NSE)", "USA (SPX)"])

if market_choice == "India (NSE)":
    target = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"])
    st.subheader(f"NSE Option Chain: {target}")
    df = get_nse_option_chain(target)
    
    if not df.empty:
        st.dataframe(df.style.background_gradient(subset=['CE_OI', 'PE_OI'], cmap='Greens'), use_container_width=True)
    else:
        st.error("Failed to fetch NSE data. The exchange might be blocking the IP.")

else:
    st.subheader("S&P 500 (SPX) Option Chain")
    session = get_browser_session()
    spx_ticker = yf.Ticker("^SPX", session=session)
    
    try:
        expiries = spx_ticker.options
        selected_expiry = st.sidebar.selectbox("Select Expiry", expiries)
        
        if st.sidebar.button("Fetch SPX Chain"):
            df = get_spx_option_chain(selected_expiry)
            if not df.empty:
                st.dataframe(df.style.background_gradient(subset=['CE_OI', 'PE_OI'], cmap='Blues'), use_container_width=True)
                
                # Simple Volatility Visualization
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df['Strike'], y=df['CE_OI'], name='Call OI'))
                fig.add_trace(go.Bar(x=df['Strike'], y=df['PE_OI'], name='Put OI'))
                fig.update_layout(title="Open Interest Distribution", barmode='group', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Yahoo Finance Rate Limit active. Please wait 1-2 minutes. Details: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Tips: Use 'Clear Cache' in the top right menu if data gets stuck.")
