# Streamlit-Based Web Dashboard for Stock Prediction

# Imports
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import xgboost as xgb

# Model registry
MODEL_REGISTRY = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# --- Cached Data Function ---
@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError("No data returned for symbol: " + symbol)
        data.dropna(inplace=True)
        data['Target'] = data['Close'].shift(-1)
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['Volatility'] = data['Close'].rolling(window=10).std()
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error retrieving data for {symbol}: {e}")
        return None

def get_headlines(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = soup.find_all("h3")
        return [h.get_text() for h in headlines if h.get_text()][:5]
    except Exception as e:
        return []

def analyze_sentiment(headlines):
    if not headlines:
        return 0.0
    sentiment_scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return np.mean(sentiment_scores)

def run_model(data, model_name):
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'Volatility']
    stock_features = data[feature_cols][:-1]
    stock_target = data['Target'][:-1]
    X_train, X_test, y_train, y_test = train_test_split(stock_features, stock_target, test_size=0.2, random_state=42)

    model = MODEL_REGISTRY[model_name]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
    else:
        importances = np.zeros(len(feature_cols))
    return y_test, predictions, rmse, r2, pd.Series(importances, index=feature_cols), pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})

def plot_predictions(y_test, pred_dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values[:50], label='Actual')
    for model_name, preds in pred_dict.items():
        ax.plot(preds[:50], label=f'Predicted ({model_name})')
    ax.set_title('Stock Price Prediction Comparison')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Stock Closing Price (USD)')
    ax.legend()
    st.pyplot(fig)

def plot_historical_prices(data, symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label='Historical Close')
    ax.set_title(f'{symbol} Historical Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Stock Insight Pro", page_icon="📈", layout="wide")
    st.title("📈 Stock Insight Pro")
    st.markdown("Compare stock predictions using different models.")

    symbols = st.text_input("Enter Stock Symbols separated by commas (e.g., TSLA, AAPL, MSFT):", value="TSLA, AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    comparison_results = []

    if st.button("Run Comparison"):
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        tabs = st.tabs(symbol_list)

        for i, symbol in enumerate(symbol_list):
            with tabs[i]:
                st.subheader(f"📉 Historical Prices for {symbol}")
                data = get_stock_data(symbol, start_date, end_date)
                if data is None:
                    continue

                plot_historical_prices(data, symbol)

                headlines = get_headlines(symbol)
                sentiment = analyze_sentiment(headlines)
                st.subheader(f"🗞️ News Sentiment Score for {symbol}: {sentiment:.2f}")

                if headlines:
                    st.markdown("**Recent Headlines:**")
                    for h in headlines:
                        st.markdown(f"- {h}")

                st.subheader("📊 Comparing Models")
                pred_dict = {}
                metrics = []
                y_test_final = None

                for model_type in MODEL_REGISTRY:
                    y_test, preds, rmse, r2, _, _ = run_model(data, model_type)
                    pred_dict[model_type] = preds
                    y_test_final = y_test
                    metrics.append({"Symbol": symbol, "Model": model_type, "RMSE": round(rmse, 2), "R2 Score": round(r2, 2)})

                plot_predictions(y_test_final, pred_dict)

                st.markdown("**Model Metrics:**")
                for m in metrics:
                    st.markdown(f"**{m['Model']}** — RMSE: {m['RMSE']}, R²: {m['R2 Score']}")
                    comparison_results.append(m)

        if comparison_results:
            st.subheader("📋 Model Comparison Summary")
            summary_df = pd.DataFrame(comparison_results)
            st.dataframe(summary_df)
            csv_summary = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Comparison Summary as CSV",
                data=csv_summary,
                file_name='comparison_summary.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
