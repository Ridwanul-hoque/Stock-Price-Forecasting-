import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Stock Price Forecasting App")

ticker = st.sidebar.selectbox(
    "Select Stock",
    options=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA"],
    format_func=lambda x: {
        "AAPL": "Apple (AAPL)",
        "GOOGL": "Google (GOOGL)",
        "MSFT": "Microsoft (MSFT)",
        "TSLA": "Tesla (TSLA)",
        "AMZN": "Amazon (AMZN)",
        "META": "Meta (META)",
        "NFLX": "Netflix (NFLX)",
        "NVDA": "NVIDIA (NVDA)"
    }[x]
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 120, 90)

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df = data[['Close']].copy()
df.columns = ['Price']
df = df.dropna()

col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${df['Price'].iloc[-1]:.2f}")
col2.metric("52-Week High", f"${df['Price'].tail(252).max():.2f}")
col3.metric("52-Week Low", f"${df['Price'].tail(252).min():.2f}")

st.subheader("Historical Closing Prices & Rolling Mean")
df['20_MA'] = df['Price'].rolling(window=20).mean()
st.line_chart(df[['Price', '20_MA']])

df_prophet = df['Price'].reset_index()
df_prophet.columns = ['ds', 'y']
df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=horizon, freq='B')
forecast = model.predict(future)

st.subheader("Forecast Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

fig = model.plot(forecast)
st.pyplot(fig)

csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)
