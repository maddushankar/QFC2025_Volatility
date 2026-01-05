
import streamlit as st

st.title("Team Collaboration App")
st.write("Hello team! This is our shared Streamlit workspace.")
import streamlit as st
import pandas as pd
from nselib import capital_market
import plotly.express as px

st.title("NSE Historical Data Explorer")

# --- INPUT SECTION ---
with st.sidebar:
    symbol = st.text_input("Enter NSE Symbol", value="SBIN").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2025-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2026-01-01"))

# --- DATA FETCHING ---
@st.cache_data
def get_historical_data(symbol, start, end):
    # nselib requires DD-MM-YYYY format
    start_str = start.strftime("%d-%m-%Y")
    end_str = end.strftime("%d-%m-%Y")
    
    # Fetch historical price and volume data
    df = capital_market.price_volume_and_deliverable_position_data(
        symbol=symbol, 
        from_date=start_str, 
        to_date=end_str
    )
    return df

if st.button("Fetch Historical Data"):
    try:
        data = get_historical_data(symbol, start_date, end_date)
        
        # Clean up columns (NSE names can have spaces/weird casing)
        data.columns = [c.strip() for c in data.columns]
        
        # Convert Date and Close Price to proper formats
        data['Date'] = pd.to_datetime(data['Date'])
        data['Close Price'] = pd.to_numeric(data['Close Price'].str.replace(',', ''))

        # --- VISUALIZATION ---
        st.subheader(f"Price History: {symbol}")
        fig = px.line(data, x='Date', y='Close Price', title=f"{symbol} Closing Prices")
        st.plotly_chart(fig, use_container_width=True)

        # --- TEAM DOWNLOAD ---
        st.success("Data loaded successfully!")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV for the Team",
            data=csv,
            file_name=f"{symbol}_historical.csv",
            mime="text/csv",
        )
        
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

# 4. Data Table
with st.expander("View Raw Data Table"):
    st.dataframe(df)
