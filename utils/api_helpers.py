import requests
import yfinance as yf
import pandas as pd
from config.settings import ALPHAVANTAGE_API_KEY, FMP_API_KEY, langfuse, trace

def fetch_yahoo_finance_data(symbols, period="1mo"):
    """Fetch stock data from Yahoo Finance API"""
    span = langfuse.span(
        name="Yahoo Finance API Call",
        parent_id=trace.id
    )
    
    try:
        data = {}
        for symbol in symbols:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            data[symbol] = {
                "price_data": hist.to_dict(),
                "info": stock.info
            }
        span.end(status="success")
        return data
    except Exception as e:
        span.end(status="error", statusMessage=str(e))
        return {"error": str(e)}

def fetch_alpha_vantage_data(symbol):
    """Fetch fundamental data from Alpha Vantage API"""
    span = langfuse.span(
        name="Alpha Vantage API Call",
        parent_id=trace.id
    )
    
    try:
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
        response = requests.get(url)
        data = response.json()
        span.end(status="success")
        return data
    except Exception as e:
        span.end(status="error", statusMessage=str(e))
        return {"error": str(e)}

def fetch_fmp_data(symbol):
    """Fetch financial statements from Financial Modeling Prep API"""
    span = langfuse.span(
        name="Financial Modeling Prep API Call",
        parent_id=trace.id
    )
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?apikey={FMP_API_KEY}"
        response = requests.get(url)
        data = response.json()
        span.end(status="success")
        return data
    except Exception as e:
        span.end(status="error", statusMessage=str(e))
        return {"error": str(e)}

def get_sp500_symbols():
    """Get a list of S&P 500 stocks"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()