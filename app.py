{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww44600\viewh25200\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Streamlit-Based Web Dashboard for Stock Prediction\
\
# Imports\
import streamlit as st\
from sklearn.datasets import load_iris\
from sklearn.model_selection import train_test_split\
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score\
import yfinance as yf\
import pandas as pd\
import numpy as np\
import matplotlib.pyplot as plt\
from textblob import TextBlob\
import requests\
from bs4 import BeautifulSoup\
\
# --- Helper Functions ---\
def get_stock_data(symbol):\
    try:\
        data = yf.download(symbol, start='2020-01-01', end='2024-12-31')\
        if data.empty:\
            raise ValueError("No data returned for symbol: " + symbol)\
        data.dropna(inplace=True)\
        data['Target'] = data['Close'].shift(-1)\
        data['MA_5'] = data['Close'].rolling(window=5).mean()\
        data['MA_10'] = data['Close'].rolling(window=10).mean()\
        data['Volatility'] = data['Close'].rolling(window=10).std()\
        data.dropna(inplace=True)\
        return data\
    except Exception as e:\
        st.error(f"Error retrieving data for \{symbol\}: \{e\}")\
        return None\
\
def get_headlines(symbol):\
    try:\
        url = f"https://finance.yahoo.com/quote/\{symbol\}/news"\
        response = requests.get(url)\
        soup = BeautifulSoup(response.text, "html.parser")\
        headlines = soup.find_all("h3")\
        return [h.get_text() for h in headlines if h.get_text()][:5]\
    except Exception as e:\
        return []\
\
def analyze_sentiment(headlines):\
    if not headlines:\
        return 0.0\
    sentiment_scores = [TextBlob(h).sentiment.polarity for h in headlines]\
    return np.mean(sentiment_scores)\
\
def run_stock_prediction(data):\
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'Volatility']\
    stock_features = data[feature_cols][:-1]\
    stock_target = data['Target'][:-1]\
    X_train, X_test, y_train, y_test = train_test_split(stock_features, stock_target, test_size=0.2, random_state=42)\
    model = RandomForestRegressor(n_estimators=100, random_state=42)\
    model.fit(X_train, y_train)\
    predictions = model.predict(X_test)\
    rmse = np.sqrt(mean_squared_error(y_test, predictions))\
    r2 = r2_score(y_test, predictions)\
    feature_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)\
    return y_test, predictions, rmse, r2, feature_importances\
\
def plot_predictions(y_test, predictions):\
    fig, ax = plt.subplots(figsize=(10, 4))\
    ax.plot(y_test.values[:50], label='Actual')\
    ax.plot(predictions[:50], label='Predicted (RF)')\
    ax.set_title('Stock Price Prediction')\
    ax.set_xlabel('Data Point Index')\
    ax.set_ylabel('Stock Closing Price (USD)')\
    ax.legend()\
    st.pyplot(fig)\
\
# --- Streamlit App ---\
def main():\
    st.title("Stock Price Prediction Dashboard")\
    symbol = st.text_input("Enter Stock Symbol (e.g., TSLA, AAPL, MSFT):", value="TSLA").upper()\
\
    if st.button("Run Prediction"):\
        data = get_stock_data(symbol)\
        if data is not None:\
            headlines = get_headlines(symbol)\
            sentiment = analyze_sentiment(headlines)\
            st.subheader(f"News Sentiment Score for \{symbol\}: \{sentiment:.2f\}")\
\
            y_test, predictions, rmse, r2, importances = run_stock_prediction(data)\
            st.subheader(f"\{symbol\} RMSE: \{rmse:.2f\}")\
            st.subheader(f"\{symbol\} R\'b2 Score: \{r2:.2f\}")\
            st.write("Feature Importances:")\
            st.bar_chart(importances)\
            plot_predictions(y_test, predictions)\
\
if __name__ == "__main__":\
    main()\
}