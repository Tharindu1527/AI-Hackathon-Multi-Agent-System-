import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import json
import yfinance as yf
from crewai import Agent, Task, Crew, Process, LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langfuse.client import Langfuse
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import nest_asyncio


# Nest_asyncio for fix asyncio event loop issues
nest_asyncio.apply()


# Load environment variables
load_dotenv()

# Setup API keys (replace with your actual keys)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.7
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

def run_stock_analysis():
    try:
        # Execute the crew workflow to get the raw result from all agents
        raw_result = stock_analysis_crew.kickoff()
        
        # Initialize the formatted result structure
        formatted_result = {
            "top_stocks": [],
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market_outlook": "",
            "sector_performance": {}
        }
        
        # Access the task outputs directly from the CrewOutput object
        tasks_output = raw_result.tasks_output
        
        # Initialize variables to store task results
        technical_analysis = {}
        fundamental_analysis = {}
        risk_assessment = {}
        portfolio_recommendation = {}
        
        # Extract data from the task outputs
        for task_output in tasks_output:
            if isinstance(task_output.raw, dict):
                if "technical_analysis" in task_output.raw:
                    technical_analysis = task_output.raw["technical_analysis"]
                elif "fundamental_analysis" in task_output.raw:
                    fundamental_analysis = task_output.raw["fundamental_analysis"]
                elif "risk_assessment" in task_output.raw:
                    risk_assessment = task_output.raw["risk_assessment"]
                elif "portfolio_recommendation" in task_output.raw:
                    portfolio_recommendation = task_output.raw["portfolio_recommendation"]
        
        # Process the portfolio recommendation data
        if portfolio_recommendation:
            formatted_result["market_outlook"] = portfolio_recommendation.get("market_outlook", 
                                                "Market outlook not provided by analysis.")
            formatted_result["sector_performance"] = portfolio_recommendation.get("sector_performance", {})
            
            recommended_stocks = portfolio_recommendation.get("recommended_stocks", [])
            for stock in recommended_stocks[:5]:
                stock_data = {
                    "symbol": stock.get("symbol"),
                    "name": stock.get("name"),
                    "technical_score": float(stock.get("technical_score", 0)),
                    "fundamental_score": float(stock.get("fundamental_score", 0)),
                    "risk_score": float(stock.get("risk_score", 0)),
                    "composite_score": float(stock.get("composite_score", 0)),
                    "recommendation": stock.get("recommendation"),
                    "target_price": float(stock.get("target_price", 0))
                }
                formatted_result["top_stocks"].append(stock_data)
        
        return formatted_result
        
    except Exception as e:
        print(f"Error in stock analysis: {str(e)}")
        raise e


def main():
    st.title("AI-Powered Stock Analysis")

    # Add a button to trigger the analysis
    if st.button("Run Stock Analysis"):
        with st.spinner("Analyzing stocks... This may take a few minutes."):
            # Fetch stock data from AI agents
            analysis_results = run_stock_analysis()

        if analysis_results:
            # Display market outlook
            st.header("Market Outlook")
            st.write(analysis_results["market_outlook"])

            # Display sector performance
            st.header("Sector Performance")
            sector_data = analysis_results["sector_performance"]
            fig = go.Figure(data=[go.Bar(x=list(sector_data.keys()), y=list(sector_data.values()))])
            fig.update_layout(title="Sector Performance", xaxis_title="Sector", yaxis_title="Performance")
            st.plotly_chart(fig)

            # Display top stocks
            st.header("Top 5 Recommended Stocks")
            for stock in analysis_results["top_stocks"]:
                with st.expander(f"{stock['name']} ({stock['symbol']})"):
                    st.markdown(f"""
                    **Technical Score:** {stock['technical_score']:.2f}  
                    **Fundamental Score:** {stock['fundamental_score']:.2f}  
                    **Risk Score:** {stock['risk_score']:.2f}  
                    **Composite Score:** {stock['composite_score']:.2f}  
                    **Recommendation:** {stock['recommendation']}  
                    **Target Price:** ${stock['target_price']:.2f}
                    """)

            # Create a comparison chart for the top stocks
            fig = go.Figure()
            for stock in analysis_results["top_stocks"]:
                fig.add_trace(go.Bar(
                    x=['Technical', 'Fundamental', 'Risk', 'Composite'],
                    y=[stock['technical_score'], stock['fundamental_score'], stock['risk_score'], stock['composite_score']],
                    name=stock['symbol']
                ))
            fig.update_layout(title="Top Stocks Comparison", barmode='group')
            st.plotly_chart(fig)

        else:
            st.warning("No stock data available. Ensure agents are responding.")

    st.sidebar.markdown("## About")
    st.sidebar.info("This AI-powered stock analysis tool uses advanced agents to analyze market data and provide investment recommendations.")

if __name__ == "__main__":
    main()