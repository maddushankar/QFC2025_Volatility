import streamlit as st
import yfinance as yf
import requests

# 1. Create a custom session to bypass rate limits
@st.cache_resource
def get_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    return session

def get_spx_data():
    session = get_yf_session()
    spx_ticker = yf.Ticker("^SPX", session=session)
    
    try:
        # Fetching options with a longer timeout and custom session
        expiries = spx_ticker.options
        return spx_ticker, expiries
    except Exception as e:
        st.error(f"Yahoo Finance is currently blocking the cloud server. Error: {e}")
        return None, []

# --- UI LOGIC ---
st.title("SPX Option Analysis")
ticker_obj, options = get_spx_data()

if options:
    expiry = st.selectbox("Select Expiry", options)
    # Your plotting logic here...
else:
    st.warning("⚠️ Rate limit hit. Try clicking 'Clear Cache' or wait 2 minutes.")
