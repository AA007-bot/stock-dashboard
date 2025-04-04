import streamlit as st
from datetime import datetime
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Redesigned Professional UI for Stock Dashboard
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the symbol or date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def display_stock_metrics(data):
    st.subheader("📊 Key Metrics")
    latest_close = float(data['Close'].iloc[-1]) if not data['Close'].empty else None
    daily_change = float(data['Close'].iloc[-1] - data['Close'].iloc[-2]) if len(data['Close']) > 1 else None
    daily_change_pct = float((daily_change / data['Close'].iloc[-2]) * 100) if daily_change is not None else None

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Close", f"${latest_close:.2f}" if latest_close is not None else "N/A")
    col2.metric("Daily Change", f"${daily_change:.2f}" if daily_change is not None else "N/A", \
                f"{daily_change_pct:.2f}%" if daily_change_pct is not None else "N/A")
    col3.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}" if not data['Volume'].empty else "N/A")

def plot_stock_chart(data, ticker):
    st.subheader(f"📈 {ticker} Price Chart")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Close Price', color='blue')
    ax.set_title(f"{ticker} Price Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

def plot_technical_indicators(data, ticker):
    st.subheader(f"📉 {ticker} Technical Indicators")
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Close'], label='Close Price', color='blue')
    ax.plot(data['SMA_50'], label='50-Day SMA', color='orange')
    ax.plot(data['SMA_200'], label='200-Day SMA', color='green')
    ax.set_title(f"{ticker} Moving Averages")
    ax.legend()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Professional Stock Dashboard", layout="wide")
    st.title("📊 Professional Stock Dashboard")

    # Sidebar for user inputs
    st.sidebar.header("Navigation")
    ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    if st.sidebar.button("Analyze Stock"):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            st.header(f"Stock Analysis for {ticker}")

            # Display key metrics
            display_stock_metrics(data)

            # Display stock chart
            plot_stock_chart(data, ticker)

            # Display technical indicators
            plot_technical_indicators(data, ticker)

if __name__ == "__main__":
    main()
