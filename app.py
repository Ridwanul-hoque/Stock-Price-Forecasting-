import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import date, timedelta

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


@st.cache_data(ttl=86400, show_spinner="Downloading stock data...")
def download_stock_data(ticker):
    end_date_download = date.today() + timedelta(days=1)

    data = yf.download(
        ticker,
        start="2000-01-01",
        end=end_date_download.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
        elif "Close" in data.columns.get_level_values(1):
            data = data.xs("Close", axis=1, level=1)
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
        else:
            return pd.DataFrame()
    elif "Close" in data.columns:
        data = data["Close"]
    else:
        return pd.DataFrame()

    data = pd.DataFrame(data)
    data.columns = ["Price"]
    data.index = pd.to_datetime(data.index)
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")
    data = data.dropna()
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    return data


if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()


try:
    all_data = download_stock_data(ticker)
except Exception:
    st.error(
        "Unable to download stock data from Yahoo Finance right now. "
        "Please try again later."
    )
    st.stop()


if all_data.empty:
    st.error("No historical data was found for the selected stock.")
    st.stop()


start_timestamp = pd.Timestamp(start_date)
end_timestamp = pd.Timestamp(end_date)

data = all_data[
    (all_data.index >= start_timestamp) &
    (all_data.index < end_timestamp)
].copy()


if data.empty:
    st.error(
        f"No trading data is available for {ticker} between "
        f"{start_date.strftime('%Y-%m-%d')} and "
        f"{end_date.strftime('%Y-%m-%d')}."
    )
    st.stop()


if len(data) < 2:
    st.error(
        "The selected date range does not contain enough trading data "
        "to generate a forecast."
    )
    st.stop()


df = data.copy()

df["20_MA"] = df["Price"].rolling(window=20, min_periods=1).mean()


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

st.dataframe(
    forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(horizon),
    use_container_width=True
)


st.subheader("Prophet Forecast")

fig = model.plot(forecast)

st.pyplot(fig, clear_figure=True)


csv = forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].to_csv(index=False)


st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)
