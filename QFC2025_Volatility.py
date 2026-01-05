import streamlit as st
from nsepython import nse_optionchain_scrapper
import pandas as pd

st.subheader("📊 Live NSE Option Chain")

# --- OPTION CHAIN FETCHING ---
@st.cache_data(ttl=60)
def get_option_chain(symbol):
    try:
        # Fetching raw payload
        payload = nse_optionchain_scrapper(symbol)
        
        # FIX: Check if 'filtered' exists before accessing it
        if payload and 'filtered' in payload:
            data = payload['filtered']['data']
        elif payload and 'records' in payload:
            # Fallback to 'records' if 'filtered' is missing
            data = payload['records']['data']
        else:
            st.error("NSE API returned an unexpected format. No 'filtered' or 'records' key found.")
            return pd.DataFrame()

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
        st.error(f"Error: {e}")
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
