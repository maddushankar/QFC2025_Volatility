
import streamlit as st

st.title("Team Collaboration App")
st.write("Hello team! This is our shared Streamlit workspace.")

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURATION & HEADERS ---
# NSE requires specific headers and a session to allow data scraping
headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9"
}

@st.cache_data(ttl=300) # Cache data for 5 minutes to avoid hitting rate limits
def fetch_nse_data(symbol="NIFTY"):
    base_url = "https://www.nseindia.com/"
    api_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    
    session = requests.Session()
    session.get(base_url, headers=headers) # Initial call to get cookies
    response = session.get(api_url, headers=headers)
    return response.json()

# --- STREAMLIT UI ---
st.title("NSE Option Chain Visualizer")

symbol = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])
raw_data = fetch_nse_data(symbol)

# 1. Extract Expiry Dates
expiry_dates = raw_data['records']['expiryDates']
selected_expiry = st.selectbox("Select Maturity Date", expiry_dates)

# 2. Process Data for Selected Expiry
data_list = []
for item in raw_data['records']['data']:
    if item['expiryDate'] == selected_expiry:
        row = {'strikePrice': item['strikePrice']}
        if 'CE' in item:
            row['CE_OI'] = item['CE']['openInterest']
        if 'PE' in item:
            row['PE_OI'] = item['PE']['openInterest']
        data_list.append(row)

df = pd.DataFrame(data_list).sort_values("strikePrice")

# 3. Plotting
st.subheader(f"Open Interest for {symbol} - {selected_expiry}")

fig = go.Figure()
fig.add_trace(go.Bar(x=df['strikePrice'], y=df['CE_OI'], name='Call OI (Resistance)', marker_color='red'))
fig.add_trace(go.Bar(x=df['strikePrice'], y=df['PE_OI'], name='Put OI (Support)', marker_color='green'))

fig.update_layout(
    barmode='group',
    xaxis_title="Strike Price",
    yaxis_title="Open Interest",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# 4. Data Table
with st.expander("View Raw Data Table"):
    st.dataframe(df)
