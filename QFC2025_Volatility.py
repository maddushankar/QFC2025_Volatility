import streamlit as st
import pandas as pd
from jugaad_data.nse import bhavcopy_fo_save
from datetime import date, datetime
import os
import plotly.express as px

st.set_page_config(page_title="NSE Historical Options Hub", layout="wide")

st.title("📊 Nifty Historical Options Data (Bhavcopy)")
st.info("Choose a historical date to download the End-of-Day (EOD) options report.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Search Filters")
    # Markets are closed on weekends; users should pick a weekday
    selected_date = st.date_input("Select Historical Date", value=date(2025, 1, 1))
    symbol = st.selectbox("Underlying Asset", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

# --- DATA PROCESSING ---
def get_historical_data(target_date):
    # Create a temporary directory to store the downloaded CSV
    tmp_path = "./data_cache"
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    
    try:
        # Downloads the FO Bhavcopy for that specific date
        file_path = bhavcopy_fo_save(target_date, tmp_path)
        df = pd.read_csv(file_path)
        
        # Clean the column names (NSE sometimes has leading/trailing spaces)
        df.columns = [c.strip() for c in df.columns]
        
        # Filter for the selected index and only 'OPTIDX' (Option Index)
        df = df[(df['SYMBOL'] == symbol) & (df['INSTRUMENT'] == 'OPTIDX')]
        return df
    except Exception as e:
        st.error(f"Could not find data for {target_date}. The market might have been closed or data is not yet available.")
        return None

# --- MAIN DASHBOARD ---
if st.button("Fetch Historical Chain"):
    raw_df = get_historical_data(selected_date)
    
    if raw_df is not None:
        # Allow user to filter by Maturity (EXPIRY_DT)
        all_expiries = sorted(raw_df['EXPIRY_DT'].unique())
        selected_expiry = st.selectbox("Select Maturity / Expiry Date", all_expiries)
        
        final_df = raw_df[raw_df['EXPIRY_DT'] == selected_expiry].copy()
        
        # Sort by strike for better visualization
        final_df = final_df.sort_values("STRIKE_PR")

        # Layout: Visualization & Table
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Open Interest Distribution - {selected_expiry}")
            # Plotting OI for Calls vs Puts
            fig = px.bar(final_df, x="STRIKE_PR", y="OPEN_INT", color="OPTION_TYP",
                         barmode="group", title="OI by Strike Price")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Raw Data Table")
            display_cols = ['STRIKE_PR', 'OPTION_TYP', 'CLOSE', 'OPEN_INT', 'CHG_IN_OI']
            st.dataframe(final_df[display_cols], height=500)
            
        # Download Link for the team
        csv = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download filtered CSV", data=csv, file_name=f"{symbol}_{selected_expiry}_data.csv")
