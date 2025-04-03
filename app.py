# Streamlit-Based Web Dashboard for Stock Prediction (Quant Edition)

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
from fpdf import FPDF
import tempfile

# Model registry
MODEL_REGISTRY = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

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
    except Exception:
        return []

def analyze_sentiment(headlines):
    if not headlines:
        return 0.0
    sentiment_scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return np.mean(sentiment_scores)

def calculate_quant_metrics(returns):
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
    std_dev = np.std(returns)
    alpha = np.mean(returns)
    beta = np.cov(returns, returns)[0][1] / (np.var(returns) + 1e-6)
    return sharpe, alpha, beta, std_dev

def backtest_strategy(predictions):
    signals = np.sign(np.diff(predictions))
    returns = signals * np.diff(predictions)
    cumulative_returns = np.cumsum(returns)
    return cumulative_returns[-1], returns, cumulative_returns

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
    returns = np.diff(predictions) / predictions[:-1]
    sharpe, alpha, beta, std_dev = calculate_quant_metrics(returns)
    strategy_return, daily_returns, cumulative_returns = backtest_strategy(predictions)

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = model.coef_
    else:
        importances = np.zeros(len(feature_cols))

    return y_test, predictions, rmse, r2, sharpe, alpha, beta, std_dev, strategy_return, cumulative_returns, pd.Series(importances, index=feature_cols), pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})

def generate_pdf_report(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Insight Pro Report", ln=True, align='C')
    pdf.ln(10)

    for entry in results:
        line = f"{entry['Symbol']} ({entry['Model']}): RMSE={entry['RMSE']}, R²={entry['R2 Score']}, Sharpe={entry['Sharpe']}, Alpha={entry['Alpha']}, Beta={entry['Beta']}, Volatility={entry['Volatility']}, StrategyReturn={entry['Strategy Return']}"
        pdf.multi_cell(0, 10, line)
        pdf.ln(2)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmpfile.name)
    return tmpfile.name

def plot_strategy_returns(cumulative_dict):
    st.subheader("📈 Cumulative Strategy Returns")
    for model, curve in cumulative_dict.items():
        plt.plot(curve, label=model)
    plt.title("Cumulative Return Based on Strategy")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

