import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date
import time

st.set_page_config(
    page_title="Stock Price Forecasting App",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Forecasting App")

ticker = st.sidebar.selectbox(
    "Select Stock",
    options=[
        "AAPL",
        "GOOGL",
        "MSFT",
        "TSLA",
        "AMZN",
        "META",
        "NFLX",
        "NVDA"
    ],
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

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2020, 1, 1),
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date(2024, 1, 1),
    min_value=date(2000, 1, 2),
    max_value=date.today()
)

horizon = st.sidebar.slider(
    "Forecast Horizon (days)",
    7,
    120,
    90
)


@st.cache_data(ttl=86400, show_spinner=False)
def download_stock_data(ticker):
    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
                multi_level_index=False
            )

            if data is not None and not data.empty:
                if "Close" in data.columns:
                    data = data[["Close"]].copy()
                    data.columns = ["Price"]

                    data.index = pd.to_datetime(data.index)

                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)

                    data["Price"] = pd.to_numeric(
                        data["Price"],
                        errors="coerce"
                    )

                    data = data.dropna(subset=["Price"])
                    data = data[~data.index.duplicated(keep="last")]
                    data = data.sort_index()

                    if not data.empty:
                        return data

            last_error = "Yahoo Finance returned no usable historical data."

        except Exception as error:
            last_error = str(error)

        if attempt < 2:
            time.sleep(2)

    return pd.DataFrame()


if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()


with st.spinner(f"Loading {ticker} historical data..."):
    data = download_stock_data(ticker)


if data.empty:
    st.error(
        f"Yahoo Finance did not return historical data for {ticker}. "
        "Please try this stock again later."
    )
    st.stop()


start_timestamp = pd.Timestamp(start_date)
end_timestamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)

df = data[
    (data.index >= start_timestamp) &
    (data.index < end_timestamp)
].copy()


if df.empty:
    available_start = data.index.min().strftime("%Y-%m-%d")
    available_end = data.index.max().strftime("%Y-%m-%d")

    st.error(
        f"No trading data is available for {ticker} in the selected date range. "
        f"Available data: {available_start} to {available_end}."
    )
    st.stop()


if len(df) < 2:
    st.error(
        "The selected date range contains insufficient historical data "
        "for forecasting. Please select a wider date range."
    )
    st.stop()


df["20_MA"] = df["Price"].rolling(
    window=20,
    min_periods=1
).mean()


col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${df['Price'].iloc[-1]:.2f}"
)

col2.metric(
    "52-Week High",
    f"${df['Price'].tail(252).max():.2f}"
)

col3.metric(
    "52-Week Low",
    f"${df['Price'].tail(252).min():.2f}"
)


st.subheader("Historical Closing Prices & Rolling Mean")

st.line_chart(
    df[["Price", "20_MA"]]
)


df_prophet = df[["Price"]].reset_index()

df_prophet.columns = ["ds", "y"]

df_prophet["ds"] = pd.to_datetime(
    df_prophet["ds"]
).dt.tz_localize(None)

df_prophet["y"] = pd.to_numeric(
    df_prophet["y"],
    errors="coerce"
)

df_prophet = df_prophet.dropna()

df_prophet = df_prophet.drop_duplicates(
    subset=["ds"]
)

df_prophet = df_prophet.sort_values("ds")


if len(df_prophet) < 2:
    st.error(
        "Not enough valid historical closing-price data is available "
        "for Prophet forecasting."
    )
    st.stop()


model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)

model.fit(df_prophet)


future = model.make_future_dataframe(
    periods=horizon,
    freq="B"
)

forecast = model.predict(future)


st.subheader("Forecast Data")

forecast_display = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].tail(horizon)

st.dataframe(
    forecast_display,
    use_container_width=True
)


st.subheader("Prophet Forecast")

fig = model.plot(forecast)

st.pyplot(
    fig,
    clear_figure=True
)


csv = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].to_csv(
    index=False
)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)
