import streamlit as st
import yfinance as yf
import pandas as pd
import requests

# --- NSE FETCHING LOGIC (Session-based) ---
def get_nse_chain(symbol):
    headers = {'user-agent': 'Mozilla/5.0'}
    base_url = "https://www.nseindia.com/option-chain"
    api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    if symbol not in ["NIFTY", "BANKNIFTY"]:
        api_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    
    session = requests.Session()
    session.get(base_url, headers=headers, timeout=10)
    response = session.get(api_url, headers=headers, timeout=10)
    
    if response.status_code == 200:
        data = response.json().get('filtered', {}).get('data', [])
        return pd.DataFrame([{"Strike": r['strikePrice'], "CE_OI": r.get('CE', {}).get('openInterest', 0), 
                              "PE_OI": r.get('PE', {}).get('openInterest', 0)} for r in data])
    return pd.DataFrame()

# --- SPX FETCHING LOGIC (yfinance) ---
def get_spx_chain(expiry):
    ticker = yf.Ticker("^SPX")
    chain = ticker.option_chain(expiry)
    # Merge Calls and Puts on Strike
    df = pd.merge(chain.calls[['strike', 'openInterest']], 
                  chain.puts[['strike', 'openInterest']], 
                  on='strike', suffixes=('_CE', '_PE'))
    return df.rename(columns={'strike': 'Strike'})

# --- APP UI ---
st.title("Global Option Volatility Dashboard")
market = st.radio("Select Market", ["NSE (India)", "SPX (USA)"])

if market == "NSE (India)":
    symbol = st.selectbox("Symbol", ["NIFTY", "BANKNIFTY", "RELIANCE"])
    if st.button("Load NSE Chain"):
        df = get_nse_chain(symbol)
        st.dataframe(df)

else:
    spx_ticker = yf.Ticker("^SPX")
    expiry = st.selectbox("Select Expiry", spx_ticker.options)
    if st.button("Load SPX Chain"):
        df = get_spx_chain(expiry)
        st.dataframe(df)
