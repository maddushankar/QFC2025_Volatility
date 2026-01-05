
import streamlit as st

st.title("Team Collaboration App")
st.write("Hello team! This is our shared Streamlit workspace.")

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIG ---
st.set_page_config(page_title="Team Finance Dashboard", layout="wide")

st.title("📈 NSE Historical Analysis (yfinance)")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Enter NSE Symbol (e.g., RELIANCE, TCS)", value="RELIANCE").upper()
    
    # Common Indices for quick access
    index_choice = st.selectbox("Or select an Index", ["None", "NIFTY 50", "BANK NIFTY"])
    
    # Mapping logic for yfinance tickers
    if index_choice == "NIFTY 50":
        ticker = "^NSEI"
    elif index_choice == "BANK NIFTY":
        ticker = "^NSEBANK"
    else:
        ticker = f"{ticker_input}.NS"

    days_back = st.slider("Past Days", min_value=30, max_value=365*5, value=365)
    start_date = datetime.now() - timedelta(days=days_back)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600) # Cache for 1 hour since historical data doesn't change often
def get_yf_data(symbol, start):
    data = yf.download(symbol, start=start)
    return data

try:
    df = get_yf_data(ticker, start_date)

    if df.empty:
        st.warning(f"No data found for {ticker}. Check the symbol and try again.")
    else:
        # --- PLOTTING ---
        st.subheader(f"Price Action for {ticker}")
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Market Data"
        )])

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            yaxis_title="Stock Price (INR)",
            height=600
        )

        
        st.plotly_chart(fig, use_container_width=True)

        # --- TEAM COLLABORATION TOOLS ---
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Raw Data Summary")
            st.dataframe(df.tail(10))
        
        with col2:
            st.write("### Team Export")
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                label="Download CSV for Reports",
                data=csv,
                file_name=f"{ticker}_history.csv",
                mime="text/csv",
            )

except Exception as e:
    st.error(f"Error fetching data: {e}")

# 4. Data Table
with st.expander("View Raw Data Table"):
    st.dataframe(df)

if st.button("Submit Changes"):
    # Perform some logic here (e.g., save to a database)
    st.success("Saved!")
    st.rerun()  # Immediately starts the script from the top
