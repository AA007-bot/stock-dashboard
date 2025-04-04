import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import xgboost as xgb
from fpdf import FPDF
import tempfile

# Streamlit-Based Web Dashboard for Stock Prediction (Quant Edition)

def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"No data found for {symbol}. Please check the symbol or date range.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def get_headlines(symbol):
    # Placeholder for fetching headlines (to be implemented)
    return ["Sample headline 1", "Sample headline 2"]

def analyze_sentiment(headlines):
    # Placeholder for sentiment analysis (to be implemented)
    return np.random.uniform(-1, 1)  # Random sentiment score for now

def run_model(data, model_name):
    # Placeholder for running models (to be implemented)
    # Returns dummy values for now
    return [None] * 12

def plot_strategy_returns(cumulative_dict):
    # Placeholder for plotting strategy returns (to be implemented)
    st.line_chart(pd.DataFrame(cumulative_dict))

def generate_pdf_report(results):
    # Placeholder for generating PDF report (to be implemented)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    return temp_file.name

def main():
    st.set_page_config(page_title="Stock Insight Pro", layout="wide")
    st.title("📊 Stock Insight Pro – Quant Dashboard")

    symbols_input = st.text_input("Enter stock symbols (comma separated):", value="TSLA, AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    if st.button("Run Analysis"):
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        all_results = []

        for symbol in symbols:
            st.subheader(f"📈 {symbol} Analysis")
            data = get_stock_data(symbol, start_date, end_date)
            if data is None:
                continue

            headlines = get_headlines(symbol)
            sentiment = analyze_sentiment(headlines)
            st.markdown(f"**News Sentiment Score**: {sentiment:.2f}")

            MODEL_REGISTRY = ["LinearRegression", "RandomForest", "XGBoost"]
            metrics_table = []
            cumulative_dict = {}
            prediction_plot_data = {}

            for model_name in MODEL_REGISTRY:
                y_test, preds, rmse, r2, sharpe, alpha, beta, std_dev, strategy_ret, cumulative_ret, importances, df_preds = run_model(data, model_name)
                cumulative_dict[model_name] = cumulative_ret
                metrics_table.append({
                    "Symbol": symbol, "Model": model_name, "RMSE": round(rmse, 2), "R2 Score": round(r2, 2),
                    "Sharpe": round(sharpe, 2), "Alpha": round(alpha, 2), "Beta": round(beta, 2),
                    "Volatility": round(std_dev, 2), "Strategy Return": round(strategy_ret, 2)
                })

            # Metrics table
            metrics_df = pd.DataFrame(metrics_table)
            st.dataframe(metrics_df)

            # Prediction preview
            st.line_chart(pd.DataFrame(prediction_plot_data))

            # Cumulative strategy return plot
            plot_strategy_returns(cumulative_dict)

            all_results.extend(metrics_table)

        if all_results:
            csv = pd.DataFrame(all_results).to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Summary", csv, "model_summary.csv", "text/csv")

            pdf_path = generate_pdf_report(all_results)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Report", f, "summary_report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
