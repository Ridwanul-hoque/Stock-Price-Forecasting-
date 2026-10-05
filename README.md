# Stock Price Forecasting App

A web-based stock price forecasting application built with Streamlit, yfinance, and Prophet. The application allows users to select a stock, define a historical date range, view historical closing prices, analyze rolling averages, and generate future price forecasts using a Prophet time-series forecasting model.

## Project Overview

The Stock Price Forecasting App provides an interactive interface for analyzing historical stock prices and generating future price predictions.

The application retrieves historical market data directly from Yahoo Finance through the yfinance library. Users can select from eight supported stocks, choose a historical date range, and specify a forecasting horizon.

The forecasting model is implemented using Prophet with weekly and yearly seasonality.

## Features

- Stock selection for eight supported companies
- Historical stock data retrieval using yfinance
- Custom start and end date selection
- Historical closing price visualization
- 20-day rolling moving average
- Current selected-period price
- 52-week high
- 52-week low
- Prophet-based stock price forecasting
- Custom forecast horizon from 7 to 120 business days
- Forecast confidence intervals
- Forecast data table
- Forecast CSV download
- Responsive Streamlit interface
- Streamlit Cloud deployment support

## Supported Stocks

The application supports the following stocks:

| Company | Ticker |
|---|---|
| Apple | AAPL |
| Google | GOOGL |
| Microsoft | MSFT |
| Tesla | TSLA |
| Amazon | AMZN |
| Meta | META |
| Netflix | NFLX |
| NVIDIA | NVDA |

## Application Screenshot

![Stock Price Forecasting App](<img width="1920" height="2330" alt="screencapture-hypgcwpr5xnmzgo5ehu2vm-streamlit-app-2026-10-05-12_49_30" src="https://github.com/user-attachments/assets/15312c8b-4436-486b-8177-9f98a03b6ed1" />
)

## Application Workflow

The application follows this workflow:

1. Select a stock from the stock dropdown.
2. Select the historical start date.
3. Select the historical end date.
4. Select the forecast horizon.
5. Click the `Load Forecast` button.
6. The application downloads historical stock data from Yahoo Finance.
7. Historical closing prices are displayed.
8. The 20-day rolling mean is calculated and displayed.
9. Key price metrics are displayed.
10. Prophet is trained using the selected historical data.
11. Future business-day prices are generated.
12. The forecast is displayed as a chart.
13. Forecast values and confidence intervals are displayed in a table.
14. The forecast data can be downloaded as a CSV file.

## Technologies Used

- Python
- Streamlit
- yfinance
- Pandas
- NumPy
- Prophet
- Matplotlib

## Project Structure

```text
stock-forecast-app/
│
├── app.py
├── requirements.txt
├── README.md
