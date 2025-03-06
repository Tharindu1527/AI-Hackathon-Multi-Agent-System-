import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import json
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langfuse.client import Langfuse

# Load environment variables
load_dotenv()

# Setup API keys (replace with your actual keys)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "demo")
FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "your_langfuse_secret_key")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "your_langfuse_public_key")

# Initialize Langfuse for telemetry
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
)

# Create a trace for the entire process
trace = langfuse.trace(
    name="Stock Analysis System",
    metadata={"timestamp": datetime.now().isoformat()}
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1,
    convert_system_message_to_human=True
)

# Define the Data Collection Agent
data_collection_agent = Agent(
    role="Data Collection Specialist",
    goal="Gather comprehensive stock data from multiple financial APIs",
    backstory="""You are a data specialist with extensive experience in financial markets.
    Your expertise lies in collecting and organizing data from various financial sources.
    You know how to query APIs efficiently and structure data for further analysis.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Define the Technical Analysis Agent
technical_analysis_agent = Agent(
    role="Technical Analysis Expert",
    goal="Perform in-depth technical analysis on stock data to identify patterns and trends",
    backstory="""You are a seasoned technical analyst with years of experience in chart patterns, 
    technical indicators, and price action analysis. You can spot trends and reversals that 
    others might miss. Your analysis is rooted in statistical evidence and historical patterns.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define the Fundamental Analysis Agent
fundamental_analysis_agent = Agent(
    role="Fundamental Analysis Expert",
    goal="Analyze company fundamentals, financial health, and news sentiment",
    backstory="""You are a fundamental analyst with a background in accounting and finance.
    You excel at dissecting financial statements, evaluating management effectiveness,
    and understanding the competitive positioning of companies. You also track news sentiment
    to gauge market perception.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define the Risk Assessment Agent
risk_assessment_agent = Agent(
    role="Risk Assessment Specialist",
    goal="Evaluate risk profiles of potential investments based on volatility and market conditions",
    backstory="""You are a risk management professional who has developed strategies for 
    major investment firms. You understand volatility, drawdowns, and correlation effects.
    Your expertise helps in balancing reward potential with risk mitigation.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define the Portfolio Recommendation Agent
portfolio_recommendation_agent = Agent(
    role="Investment Advisor",
    goal="Synthesize all analyses to recommend the top 5 investable stocks",
    backstory="""You are a senior investment advisor who has guided high-net-worth clients
    through multiple market cycles. You have a holistic view of the market and can weigh
    different analytical perspectives to form a coherent investment strategy. You focus on
    identifying the best opportunities with favorable risk-reward profiles.""",
    verbose=True,
    allow_delegation=True,
    llm=llm
)

# Helper functions for API calls
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

# Define Tasks

# Task 1: Collect Data
collect_data_task = Task(
    description="""
    Collect comprehensive stock data for potential analysis. Follow these steps:
    1. Get a list of the top 30 stocks by market cap in the S&P 500 index
    2. For each stock, fetch data from:
       - Yahoo Finance: price, volume, and basic info
       - Alpha Vantage: company fundamentals
       - Financial Modeling Prep: financial statements
    3. Organize the data in a structured format for further analysis
    4. Return a JSON with the collected data
    
    Example stocks to include: AAPL, MSFT, AMZN, GOOGL, META
    
    The data should include at minimum:
    - Current and historical prices (1 month)
    - Key financial metrics (P/E, EPS, dividend yield)
    - Company overview
    
    Your output should be detailed enough for technical and fundamental analysis.
    """,
    agent=data_collection_agent,
    expected_output="""A detailed JSON containing structured data from all three APIs for at least 30 major stocks, 
    ready for analysis by other agents. Ensure it includes historical prices, key financial metrics, 
    and fundamental information."""
)

# Task 2: Technical Analysis
technical_analysis_task = Task(
    description="""
    Perform comprehensive technical analysis on the collected stock data. Your analysis should include:
    
    1. Trend analysis using moving averages (50-day and 200-day)
    2. Momentum indicators assessment (RSI, MACD)
    3. Volume analysis and what it indicates about price movements
    4. Support and resistance levels identification
    5. Chart pattern recognition
    
    For each stock, provide:
    - Current trend direction (bullish, bearish, or sideways)
    - Key technical indicators and their current readings
    - Technical strength score (1-10)
    - Potential price targets based on chart patterns
    - Any warning signals or confirmation signals
    
    Rank the stocks based on their technical strength and provide justification for your rankings.
    Identify the top 10 stocks with the strongest technical setups.
    """,
    agent=technical_analysis_agent,
    expected_output="""A detailed technical analysis report for each stock, including trend analysis, 
    indicator readings, support/resistance levels, and a final ranking of the top 10 stocks based on 
    technical strength with clear justification.""",
    context=[collect_data_task]
)

# Task 3: Fundamental Analysis
fundamental_analysis_task = Task(
    description="""
    Analyze the fundamental health and outlook of each stock using the collected data. Your analysis should include:
    
    1. Profitability metrics assessment (ROE, ROA, profit margins)
    2. Valuation analysis (P/E, P/B, P/S ratios) relative to industry and historical averages
    3. Growth prospects evaluation based on historical performance and forward guidance
    4. Balance sheet strength and debt levels
    5. Dividend policy and sustainability
    6. Recent news sentiment and its impact on future prospects
    
    For each stock, provide:
    - Overall fundamental health grade (A to F)
    - Key strengths and weaknesses
    - Valuation assessment (undervalued, fairly valued, overvalued)
    - Growth outlook (poor, moderate, strong)
    - Recent news sentiment summary
    
    Rank the stocks based on their fundamental attractiveness and provide justification for your rankings.
    Identify the top 10 stocks with the strongest fundamentals.
    """,
    agent=fundamental_analysis_agent,
    expected_output="""A comprehensive fundamental analysis for each stock, including profitability, 
    valuation, growth prospects, balance sheet analysis, and a final ranking of the top 10 stocks based 
    on fundamental strength with clear justification.""",
    context=[collect_data_task]
)

# Task 4: Risk Assessment
risk_assessment_task = Task(
    description="""
    Evaluate the risk profile of each stock based on quantitative and qualitative factors. Your assessment should include:
    
    1. Volatility analysis (Beta, standard deviation of returns)
    2. Drawdown analysis (maximum historical drawdowns)
    3. Liquidity assessment (trading volume, bid-ask spreads)
    4. Industry and macroeconomic risk factors
    5. Company-specific risks (competition, regulatory, litigation)
    
    For each stock, provide:
    - Overall risk score (1-10, where 1 is lowest risk and 10 is highest)
    - Volatility metrics and what they indicate
    - Maximum drawdown potential in different market scenarios
    - Key risk factors specific to the company
    - Risk mitigation recommendations
    
    Rank the stocks based on their risk-adjusted return potential and provide justification for your rankings.
    Identify the 10 stocks with the most favorable risk-reward profiles.
    """,
    agent=risk_assessment_agent,
    expected_output="""A detailed risk assessment for each stock, including volatility metrics, 
    drawdown analysis, liquidity assessment, key risk factors, and a final ranking of the top 10 stocks 
    with the most favorable risk-reward profiles.""",
    context=[collect_data_task, technical_analysis_task, fundamental_analysis_task]
)

# Task 5: Final Portfolio Recommendation
portfolio_recommendation_task = Task(
    description="""
    Synthesize all previous analyses to identify the top 5 investable stocks in the US market for today.
    
    Your recommendation should:
    1. Integrate technical, fundamental, and risk analyses
    2. Consider current market conditions and sector trends
    3. Balance growth potential with risk mitigation
    4. Include near-term catalysts and potential headwinds
    
    For each recommended stock, provide:
    - A comprehensive investment thesis
    - Why it ranks in the top 5
    - Key metrics that support the recommendation
    - Suggested position sizing based on risk profile
    - Potential entry points and price targets
    - Recommended holding period
    
    Your final output should be a professional investment recommendation report that could be presented to clients.
    """,
    agent=portfolio_recommendation_agent,
    expected_output="""A professional investment recommendation report identifying the top 5 investable 
    stocks with comprehensive justification, supported by technical, fundamental, and risk analyses. 
    Include specific entry points, price targets, and holding periods.""",
    context=[technical_analysis_task, fundamental_analysis_task, risk_assessment_task]
)

# Create the Crew
stock_analysis_crew = Crew(
    agents=[
        data_collection_agent,
        technical_analysis_agent,
        fundamental_analysis_agent,
        risk_assessment_agent,
        portfolio_recommendation_agent
    ],
    tasks=[
        collect_data_task,
        technical_analysis_task,
        fundamental_analysis_task,
        risk_assessment_task,
        portfolio_recommendation_task
    ],
    verbose=True,
    process=Process.sequential
)

# Streamlit UI
def main():
    st.set_page_config(page_title="Multi-Agent Stock Analysis System", layout="wide")
    
    st.title("Multi-Agent Stock Analysis System")
    st.write("Powered by CrewAI and Google Gemini 1.5")
    
    with st.sidebar:
        st.header("System Control")
        if st.button("Run Full Analysis"):
            with st.spinner("Agents are working on your analysis..."):
                # Execute the crew
                result = stock_analysis_crew.kickoff()
                
                # Save result to session state
                st.session_state.result = result
    
    if 'result' in st.session_state:
        st.header("Top 5 Investable Stocks")
        st.write(st.session_state.result)
        
        # Add visualizations based on the results
        # This is a placeholder - you would parse the result and create appropriate visualizations
        st.subheader("Performance Comparison")
        data = {
            'Stock': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'],
            'Technical Score': [8.5, 9.0, 7.8, 8.2, 7.5],
            'Fundamental Score': [8.0, 9.2, 7.5, 8.5, 7.0],
            'Risk Score': [3.5, 4.0, 5.2, 4.5, 6.0]
        }
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index('Stock'))
        
        # Display detailed analysis for each recommended stock
        st.subheader("Detailed Analysis")
        tabs = st.tabs(['Technical', 'Fundamental', 'Risk', 'Final Recommendation'])
        
        with tabs[0]:
            st.write("Technical Analysis Results")
            # Display technical analysis details
            
        with tabs[1]:
            st.write("Fundamental Analysis Results")
            # Display fundamental analysis details
            
        with tabs[2]:
            st.write("Risk Assessment Results")
            # Display risk assessment details
            
        with tabs[3]:
            st.write("Final Portfolio Recommendations")
            # Display final recommendations
    
    st.sidebar.header("Telemetry")
    st.sidebar.write("System telemetry is being tracked with Langfuse")
    
    st.sidebar.header("About")
    st.sidebar.write("""
    This multi-agent system uses 5 specialized agents to analyze stock market data
    and identify the top 5 investable stocks in the US market. The system integrates
    data from Yahoo Finance, Alpha Vantage, and Financial Modeling Prep APIs.
    """)

if __name__ == "__main__":
    main()