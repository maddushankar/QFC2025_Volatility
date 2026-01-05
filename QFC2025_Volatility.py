import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Team Finance Dashboard", layout="wide")

st.title("📈 NSE Historical Analysis")
st.write("Hello team! This is our shared workspace for Volatility analysis.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Enter NSE Stock Symbol", value="RELIANCE").upper()
    index_choice = st.selectbox("Or select an Index", ["None", "NIFTY 50", "BANK NIFTY"])
    
    # Corrected Mapping Logic
    if index_choice == "NIFTY 50":
        ticker = "^NSEI"
    elif index_choice == "BANK NIFTY":
        ticker = "^NSEBANK"
    else:
        # If 'None' is selected, use the manual text input
        ticker = f"{ticker_input}.NS"

    days_back = st.slider("Past Days", min_value=30, max_value=365*5, value=365)
    start_date = datetime.now() - timedelta(days=days_back)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_yf_data(symbol, start):
    # Fetching data using yfinance
    data = yf.download(symbol, start=start)
    return data

# --- MAIN DISPLAY ---
try:
    df = get_yf_data(ticker, start_date)

    if df.empty:
        st.warning(f"⚠️ No data found for {ticker}. Check the symbol naming.")
    else:
        st.subheader(f"Price Action for {ticker}")
        
        # Create Candlestick Chart
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
            yaxis_title="Price (INR)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Team Data Tools
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Recent Data (Last 5 Days)")
            st.dataframe(df.tail(5))
        
        with col2:
            st.write("### Team Export")
            csv = df.to_csv().encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name=f"{ticker}_data.csv")

except Exception as e:
    st.error(f"Error fetching data: {e}")

# Bottom refresh button for the team
if st.button("🔄 Force Refresh Cache"):
    get_yf_data.clear()
    st.rerun()
