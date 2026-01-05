import requests
import pandas as pd
import streamlit as st

@st.cache_data(ttl=60)
def get_option_chain(symbol):
    # 1. Mimic a real browser
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
    }
    
    # 2. NSE requires you to visit the main page first to get 'cookies'
    base_url = "https://www.nseindia.com/option-chain"
    api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    
    # Use NIFTY/BANKNIFTY for indices, but different URL for stocks
    if symbol not in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        api_url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

    try:
        session = requests.Session()
        # Hit the home page first to set cookies
        session.get(base_url, headers=headers, timeout=10)
        
        # Now fetch the actual data
        response = session.get(api_url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            st.error(f"NSE denied access (Status {response.status_code}). Try again in 5 mins.")
            return pd.DataFrame()

        payload = response.json()

        # Check for the keys properly
        if 'filtered' in payload:
            data = payload['filtered']['data']
        elif 'records' in payload:
            data = payload['records']['data']
        else:
            st.warning("Data found but structure is empty. Market might be closed.")
            return pd.DataFrame()

        # Process into DataFrame
        chain_list = []
        for row in data:
            strike = row.get('strikePrice')
            ce = row.get('CE', {})
            pe = row.get('PE', {})
            chain_list.append({
                "PE_OI": pe.get('openInterest', 0),
                "PE_LTP": pe.get('lastPrice', 0),
                "Strike": strike,
                "CE_LTP": ce.get('lastPrice', 0),
                "CE_OI": ce.get('openInterest', 0)
            })
        return pd.DataFrame(chain_list)

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# --- DISPLAY LOGIC ---
target_symbol = st.selectbox("Select Underlying", ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS"])

if st.button("Load Option Chain"):
    df_chain = get_option_chain(target_symbol)
    
    if not df_chain.empty:
        # Highlight the At-The-Money (ATM) strike roughly
        st.write(f"Showing Option Chain for {target_symbol}")
        
        # Basic formatting to look like a trading terminal
        st.dataframe(
            df_chain.style.background_gradient(subset=['CE_OI', 'PE_OI'], cmap='Greens'),
            use_container_width=True
        )
        
        # --- VOLATILITY ANALYSIS ---
        st.info("💡 Heavy OI in CE usually acts as resistance; heavy PE OI acts as support.")
