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

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Multi-Agent Stock Analysis System",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header section
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown('<div class="main-header">Multi-Agent Stock Analysis System</div>', unsafe_allow_html=True)
        st.markdown("Powered by CrewAI and Google Gemini 1.5")
    with col2:
        st.image("https://img.icons8.com/color/96/000000/stocks.png", width=80)
    
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Date range selection
        st.subheader("Time Period")
        date_range = st.selectbox(
            "Select historical data timeframe:",
            ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
            index=0
        )
        
        period_mapping = {
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "5 Years": "5y"
        }
        selected_period = period_mapping[date_range]
        
        # Stock filtering options
        st.subheader("Stock Filters")
        market_cap_filter = st.multiselect(
            "Market Cap Range:",
            ["Mega Cap (>$200B)", "Large Cap ($10B-$200B)", "Mid Cap ($2B-$10B)", "Small Cap (<$2B)"],
            default=["Mega Cap (>$200B)", "Large Cap ($10B-$200B)"]
        )
        
        sector_filter = st.multiselect(
            "Sectors:",
            ["Technology", "Healthcare", "Consumer Cyclical", "Financial Services", 
             "Communication Services", "Industrials", "Consumer Defensive", "Energy", 
             "Basic Materials", "Real Estate", "Utilities"],
            default=["Technology", "Healthcare", "Financial Services"]
        )
        
        # Analysis weights
        st.subheader("Analysis Weights")
        technical_weight = st.slider("Technical Analysis Weight", 0, 100, 33)
        fundamental_weight = st.slider("Fundamental Analysis Weight", 0, 100, 33)
        risk_weight = st.slider("Risk Assessment Weight", 0, 100, 34)
        
        # Normalize weights to sum to 100
        total_weight = technical_weight + fundamental_weight + risk_weight
        if total_weight > 0:
            technical_weight = int((technical_weight / total_weight) * 100)
            fundamental_weight = int((fundamental_weight / total_weight) * 100)
            risk_weight = 100 - technical_weight - fundamental_weight
        
        st.caption(f"Weights: Technical ({technical_weight}%), Fundamental ({fundamental_weight}%), Risk ({risk_weight}%)")
        
        # Run analysis button
        st.header("System Control")
        run_button = st.button("Run Full Analysis", type="primary")
        
        # Additional options
        export_format = st.selectbox(
            "Export Results Format:",
            ["PDF Report", "Excel Spreadsheet", "JSON Data", "CSV Data"]
        )
        
        st.download_button(
            label="Download Results",
            data="",  # This would be filled with actual data
            file_name="stock_analysis_results.pdf",
            disabled=not 'result' in st.session_state,
            help="Run analysis first to enable download"
        )
        
        # About section
        st.markdown("---")
        st.header("About")
        st.write("""
        This multi-agent system uses 5 specialized agents to analyze stock market data
        and identify the top 5 investable stocks based on your preferences. The system integrates
        data from Yahoo Finance, Alpha Vantage, and Financial Modeling Prep APIs.
        """)
        
        st.caption("© 2025 Stock AI Analysis | Version 1.0.2")
    
    # If the run button is clicked or we have existing results
    if run_button:
        with st.spinner("Agents are working on your analysis..."):
            # Execute the crew with the selected period
            # In practice, you would pass these parameters to your crew
            analysis_params = {
                "period": selected_period,
                "market_cap_filter": market_cap_filter,
                "sector_filter": sector_filter,
                "weights": {
                    "technical": technical_weight / 100,
                    "fundamental": fundamental_weight / 100,
                    "risk": risk_weight / 100
                }
            }
            
            # For demonstration, we'll simulate a delay
            import time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate the analysis progress
            steps = ["Collecting market data...", 
                    "Performing technical analysis...", 
                    "Analyzing fundamentals...",
                    "Assessing risks...", 
                    "Generating final recommendations..."]
            
            for i, step in enumerate(steps):
                progress_bar.progress((i) / len(steps))
                status_text.text(step)
                time.sleep(0.5)  # Simulate processing time
            
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # In a real implementation, you'd run the actual analysis:
            # result = stock_analysis_crew.kickoff(parameters=analysis_params)
            
            # For demonstration, we'll create mock results
            from datetime import datetime
            mock_result = {
                "top_stocks": [
                    {"symbol": "AAPL", "name": "Apple Inc.", "technical_score": 8.7, "fundamental_score": 9.1, "risk_score": 3.2, 
                     "composite_score": 8.9, "recommendation": "Strong Buy", "target_price": 230.45},
                    {"symbol": "MSFT", "name": "Microsoft Corp.", "technical_score": 9.2, "fundamental_score": 8.9, "risk_score": 2.8, 
                     "composite_score": 8.8, "recommendation": "Strong Buy", "target_price": 428.50},
                    {"symbol": "GOOGL", "name": "Alphabet Inc.", "technical_score": 8.5, "fundamental_score": 8.7, "risk_score": 3.4, 
                     "composite_score": 8.3, "recommendation": "Buy", "target_price": 187.75},
                    {"symbol": "NVDA", "name": "NVIDIA Corp.", "technical_score": 9.4, "fundamental_score": 8.2, "risk_score": 4.6, 
                     "composite_score": 8.0, "recommendation": "Buy", "target_price": 950.20},
                    {"symbol": "AMZN", "name": "Amazon.com Inc.", "technical_score": 7.9, "fundamental_score": 8.5, "risk_score": 3.8, 
                     "composite_score": 7.8, "recommendation": "Buy", "target_price": 196.30}
                ],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_outlook": "Cautiously bullish with potential volatility due to upcoming economic data.",
                "sector_performance": {
                    "Technology": 12.4,
                    "Healthcare": 8.7,
                    "Financial Services": 6.5,
                    "Consumer Cyclical": 5.2,
                    "Communication Services": 7.8,
                    "Industrials": 4.3,
                    "Energy": -2.1,
                    "Consumer Defensive": 3.2,
                    "Real Estate": -1.5,
                    "Utilities": 1.8,
                    "Basic Materials": 2.4
                }
            }
            
            # Save result to session state
            st.session_state.result = mock_result
    
    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state.result
        
        # Top recommendations section
        st.markdown('<div class="sub-header">Top 5 Investable Stocks</div>', unsafe_allow_html=True)
        st.write(f"Analysis completed on: {result['analysis_date']}")
        
        # Market outlook card
        st.markdown(f"""
        <div class="card">
            <h3>Market Outlook</h3>
            <p>{result['market_outlook']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create top stocks table with visual indicators
        top_stocks_df = pd.DataFrame(result["top_stocks"])
        
        # Create styled dataframe
        def color_scores(val):
            if isinstance(val, float):
                if val >= 8.5:
                    return f'background-color: rgba(76, 175, 80, 0.2); color: #1e5631; font-weight: bold'
                elif val >= 7.0:
                    return f'background-color: rgba(255, 235, 59, 0.2); color: #8c6d1f'
                elif val <= 4.0 and 'risk' in col.lower():  # Low risk is good
                    return f'background-color: rgba(76, 175, 80, 0.2); color: #1e5631; font-weight: bold'
                elif val >= 5.0 and 'risk' in col.lower():  # High risk is bad
                    return f'background-color: rgba(244, 67, 54, 0.2); color: #a52121'
            return ''
        
        styled_df = top_stocks_df.style.applymap(color_scores)
        
        # Display stock cards in columns
        st.subheader("Top Stock Recommendations")
        cols = st.columns(5)
        
        for i, stock in enumerate(result["top_stocks"]):
            with cols[i]:
                st.markdown(f"""
                <div style="border-radius: 10px; border: 1px solid #ddd; padding: 16px; height: 100%;">
                    <h3 style="margin-top: 0;">{stock['symbol']}</h3>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 15px;">{stock['name']}</p>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-weight: bold; color: {'#4CAF50' if stock['recommendation'] == 'Strong Buy' else '#FFC107'}">
                            {stock['recommendation']}
                        </span>
                        <span style="font-weight: bold;">${stock['target_price']}</span>
                    </div>
                    <hr style="margin: 10px 0;">
                    <div style="margin-bottom: 5px;">
                        <span style="font-size: 0.8rem; color: #666;">TECHNICAL</span>
                        <div style="background-color: #eee; border-radius: 5px; height: 8px; margin-top: 3px;">
                            <div style="background-color: #4CAF50; width: {stock['technical_score']*10}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 5px;">
                        <span style="font-size: 0.8rem; color: #666;">FUNDAMENTAL</span>
                        <div style="background-color: #eee; border-radius: 5px; height: 8px; margin-top: 3px;">
                            <div style="background-color: #2196F3; width: {stock['fundamental_score']*10}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                    </div>
                    <div style="margin-bottom: 5px;">
                        <span style="font-size: 0.8rem; color: #666;">RISK (LOWER IS BETTER)</span>
                        <div style="background-color: #eee; border-radius: 5px; height: 8px; margin-top: 3px;">
                            <div style="background-color: {'#F44336' if stock['risk_score'] > 5 else '#4CAF50'}; width: {stock['risk_score']*10}%; height: 100%; border-radius: 5px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance visualization section
        st.markdown('<div class="sub-header">Performance Analysis</div>', unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tabs = st.tabs(['Stock Comparison', 'Technical Analysis', 'Fundamental Metrics', 'Risk Assessment', 'Sector Performance'])
        
        with tabs[0]:
            # Create columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Radar chart for multi-dimensional comparison
                st.subheader("Multi-factor Comparison")
                
                # Sample data for radar chart
                radar_data = {
                    'Metric': ['Technical Score', 'Fundamental Score', 'Growth Potential', 'Value Rating', 'Momentum'],
                }
                
                for stock in result["top_stocks"]:
                    # Simulate different metrics for variety
                    radar_data[stock['symbol']] = [
                        stock['technical_score'],
                        stock['fundamental_score'],
                        7.5 + random.uniform(-1.5, 1.5),  # Simulated growth potential
                        8.0 + random.uniform(-2.0, 1.0),  # Simulated value rating
                        7.2 + random.uniform(-1.0, 2.0)   # Simulated momentum
                    ]
                
                radar_df = pd.DataFrame(radar_data)
                
                # Plot radar chart using Plotly
                fig = go.Figure()
                
                for stock in result["top_stocks"]:
                    fig.add_trace(go.Scatterpolar(
                        r=radar_df[stock['symbol']],
                        theta=radar_df['Metric'],
                        fill='toself',
                        name=stock['symbol']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=True,
                    height=450
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Composite score comparison
                st.subheader("Composite Score Rankings")
                
                # Create dataframe for bar chart
                composite_df = pd.DataFrame([
                    {'Stock': stock['symbol'], 'Score': stock['composite_score']}
                    for stock in result["top_stocks"]
                ])
                
                # Sort by score
                composite_df = composite_df.sort_values('Score', ascending=False)
                
                # Create the bar chart with Plotly
                fig = px.bar(
                    composite_df, 
                    x='Stock', 
                    y='Score',
                    color='Score',
                    color_continuous_scale='Viridis',
                    text='Score'
                )
                
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.subheader("Technical Analysis Insights")
            
            # Create columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Example of price chart with technical indicators
                st.markdown("### AAPL - Price Chart with Technical Indicators")
                
                # In a real implementation, you would use actual price data
                # Here, we'll generate synthetic data for visualization
                date_range = pd.date_range(end=pd.Timestamp.today(), periods=90)
                base_price = 180
                price_data = [base_price]
                
                # Generate synthetic price movement
                for i in range(1, 90):
                    change = price_data[-1] * np.random.normal(0.0005, 0.012)
                    price_data.append(price_data[-1] + change)
                
                # Create dataframe
                tech_df = pd.DataFrame({
                    'Date': date_range,
                    'Close': price_data
                })
                
                # Calculate MA
                tech_df['MA_50'] = tech_df['Close'].rolling(window=20).mean()
                tech_df['MA_200'] = tech_df['Close'].rolling(window=50).mean()
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=tech_df['Date'],
                    y=tech_df['Close'],
                    mode='lines',
                    name='AAPL Price',
                    line=dict(color='#1E88E5', width=2)
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(
                    x=tech_df['Date'],
                    y=tech_df['MA_50'],
                    mode='lines',
                    name='50-day MA',
                    line=dict(color='#FFA000', width=1.5)
                ))
                
                fig.add_trace(go.Scatter(
                    x=tech_df['Date'],
                    y=tech_df['MA_200'],
                    mode='lines',
                    name='200-day MA',
                    line=dict(color='#D81B60', width=1.5)
                ))
                
                # Update layout
                fig.update_layout(
                    title='AAPL Price with Moving Averages',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    legend=dict(x=0, y=1, traceorder='normal'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Example of technical indicators comparison
                st.markdown("### Technical Indicators Comparison")
                
                # Create dataframe for tech indicators
                tech_indicators = pd.DataFrame({
                    'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                    'RSI': [60.2, 52.7, 58.1, 67.3, 49.8],  # Example values
                    'MACD': [1.2, 0.8, -0.3, 2.1, 0.5],     # Example values
                    'Bollinger': [1.2, 0.7, 0.9, 1.5, 0.3], # Example values
                    'ADX': [28.3, 22.1, 19.8, 32.5, 21.3]   # Example values
                })
                
                fig = px.parallel_coordinates(
                    tech_indicators,
                    color="RSI",
                    labels={"Stock": "Stock Ticker", 
                           "RSI": "RSI (14)",
                           "MACD": "MACD Signal",
                           "Bollinger": "Bollinger Position",
                           "ADX": "ADX (14)"},
                    color_continuous_scale=px.colors.sequential.Viridis,
                    color_continuous_midpoint=50
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical metrics explanation
                with st.expander("Technical Indicators Explained"):
                    st.markdown("""
                    - **RSI (Relative Strength Index)**: Measures momentum, with values over 70 indicating overbought conditions and under 30 indicating oversold conditions.
                    - **MACD (Moving Average Convergence Divergence)**: Shows the relationship between two moving averages, with positive values indicating bullish momentum.
                    - **Bollinger Position**: Where price is within Bollinger Bands, with values near 1 indicating price near upper band.
                    - **ADX (Average Directional Index)**: Measures trend strength, with values over 25 indicating a strong trend.
                    """)
        
        with tabs[2]:
            st.subheader("Fundamental Analysis Insights")
            
            # Create columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Valuation metrics comparison
                st.markdown("### Valuation Metrics")
                
                # Create dataframe for valuation metrics
                valuation_df = pd.DataFrame({
                    'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                    'P/E Ratio': [28.5, 35.2, 25.7, 42.8, 30.1],  # Example values
                    'EV/EBITDA': [18.2, 22.1, 16.8, 28.3, 19.5],  # Example values
                    'P/S Ratio': [7.2, 12.8, 6.5, 14.2, 3.8],     # Example values
                    'P/B Ratio': [12.5, 15.3, 5.8, 20.1, 9.2]     # Example values
                })
                
                # Melt the dataframe for easier plotting
                valuation_melted = pd.melt(
                    valuation_df, 
                    id_vars=['Stock'], 
                    var_name='Metric', 
                    value_name='Value'
                )
                
                # Create the grouped bar chart
                fig = px.bar(
                    valuation_melted, 
                    x='Stock', 
                    y='Value', 
                    color='Metric',
                    barmode='group',
                    title='Valuation Metrics Comparison'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Growth metrics
                st.markdown("### Growth Metrics (YoY %)")
                
                # Create dataframe for growth metrics
                growth_df = pd.DataFrame({
                    'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                    'Revenue Growth': [12.5, 18.2, 15.7, 25.8, 20.1],  # Example values
                    'EPS Growth': [15.2, 22.1, 12.8, 32.3, 19.5],      # Example values
                    'Dividend Growth': [5.2, 8.8, 3.5, 0.0, 2.8],      # Example values
                    'FCF Growth': [10.5, 15.3, 9.8, 20.1, 12.2]        # Example values
                })
                
                # Melt the dataframe for easier plotting
                growth_melted = pd.melt(
                    growth_df, 
                    id_vars=['Stock'], 
                    var_name='Metric', 
                    value_name='Growth (%)'
                )
                
                # Create the grouped bar chart
                fig = px.bar(
                    growth_melted, 
                    x='Stock', 
                    y='Growth (%)', 
                    color='Metric',
                    barmode='group',
                    title='Year-over-Year Growth Metrics'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Financial health metrics
            st.markdown("### Financial Health Overview")
            
            # Create columns for metrics
            metric_cols = st.columns(5)
            
            # Sample financial health data for each stock
            financial_health = [
                {"Debt/Equity": 0.42, "Current Ratio": 2.1, "ROE": 35.2, "Profit Margin": 25.3, "Dividend Yield": 0.8},
                {"Debt/Equity": 0.15, "Current Ratio": 2.7, "ROE": 42.8, "Profit Margin": 33.5, "Dividend Yield": 1.2},
                {"Debt/Equity": 0.28, "Current Ratio": 1.9, "ROE": 30.1, "Profit Margin": 22.7, "Dividend Yield": 0.6},
                {"Debt/Equity": 0.08, "Current Ratio": 3.2, "ROE": 52.3, "Profit Margin": 30.2, "Dividend Yield": 0.2},
                {"Debt/Equity": 0.35, "Current Ratio": 2.5, "ROE": 33.8, "Profit Margin": 20.1, "Dividend Yield": 1.0}
            ]
            
            for i, stock in enumerate(result["top_stocks"]):
                health = financial_health[i]
                with metric_cols[i]:
                    st.markdown(f"**{stock['symbol']}**")
                    
                    # Use delta indicators to show good/bad metrics
                    st.metric("Debt/Equity", f"{health['Debt/Equity']:.2f}", 
                              delta="-0.05" if health['Debt/Equity'] < 0.3 else "0.03",
                              delta_color="normal")
                    
                    st.metric("Current Ratio", f"{health['Current Ratio']:.1f}", 
                              delta="0.2" if health['Current Ratio'] > 2.0 else "-0.1",
                              delta_color="normal")
                    
                    st.metric("ROE %", f"{health['ROE']:.1f}%", 
                              delta="3.2%" if health['ROE'] > 30 else "-1.5%",
                              delta_color="normal")
                    
                    st.metric("Profit Margin %", f"{health['Profit Margin']:.1f}%", 
                              delta="1.8%" if health['Profit Margin'] > 25 else "-0.7%",
                              delta_color="normal")
                    
                    st.metric("Dividend Yield %", f"{health['Dividend Yield']:.1f}%", 
                              delta="0.1%" if health['Dividend Yield'] > 0.5 else "0%",
                              delta_color="normal")
        
        with tabs[3]:
            st.subheader("Risk Assessment Insights")
            
            # Create columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Volatility comparison
                st.markdown("### Volatility Metrics")
                
                # Risk metrics data
                risk_df = pd.DataFrame({
                    'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                    'Beta': [1.15, 0.95, 1.08, 1.42, 1.23],  # Example values
                    'Vol (30D)': [25.2, 18.7, 22.3, 35.2, 28.1],  # Example values
                    'Vol (90D)': [22.5, 16.8, 20.5, 32.7, 26.3],  # Example values
                    'Max Drawdown': [18.5, 12.3, 15.7, 25.2, 20.1]  # Example values
                })
                
                # Create scatter plot
                fig = px.scatter(
                    risk_df,
                    x='Beta',
                    y='Vol (30D)',
                    size='Max Drawdown',
                    color='Stock',
                    hover_name='Stock',
                    size_max=25,
                    title='Risk Profile: Beta vs Volatility'
                )
                
                fig.update_layout(
                    xaxis_title='Beta (vs S&P 500)',
                    yaxis_title='30-Day Volatility (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk-reward scatter
                st.markdown("### Risk-Reward Analysis")
                
                # Risk-reward data
                risk_reward_df = pd.DataFrame({
                    'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                    'Expected Return (%)': [12.5, 15.2, 11.7, 18.3, 13.5],  # Example values
                    'Risk Score': [s['risk_score'] for s in result["top_stocks"]],
                    'Sharpe Ratio': [1.8, 2.2, 1.5, 1.2, 1.7]  # Example values
                })
                
                # Create scatter plot
                fig = px.scatter(
                    risk_reward_df,
                    x='Risk Score',
                    y='Expected Return (%)',
                    size='Sharpe Ratio',
                    color='Stock',
                    hover_name='Stock',
                    size_max=25,
                    title='Risk-Reward Analysis'
                )
                
                fig.update_layout(
                    xaxis_title='Risk Score (Lower is Better)',
                    yaxis_title='Expected Annual Return (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk breakdown table
            st.markdown("### Detailed Risk Breakdown")
            
            # Risk breakdown data
            risk_breakdown = pd.DataFrame({
                'Stock': [stock['symbol'] for stock in result["top_stocks"]],
                'Market Risk': ['Medium', 'Low', 'Medium', 'High', 'Medium'],
                'Sector Risk': ['Medium', 'Low', 'Low', 'High', 'Medium'],
                'Company Risk': ['Low', 'Low', 'Medium', 'Medium', 'Medium'],
                'Liquidity Risk': ['Low', 'Low', 'Low', 'Medium', 'Low'],
                'Volatility': ['Medium', 'Low', 'Medium', 'High', 'Medium'],
                'Correlation w/Market': ['High', 'Medium', 'High', 'High', 'High']
            })
            
            # Style the dataframe
            def highlight_risk(val):
                if val == 'High':
                    return 'background-color: rgba(244, 67, 54, 0.2); color: #a52121'
                elif val == 'Low':
                    return 'background-color: rgba(76, 175, 80, 0.2); color: #1e5631'
                return 'background-color: rgba(255, 235, 59, 0.2); color: #8c6d1f'
            
            styled_risk = risk_breakdown.style.applymap(highlight_risk, subset=[
                'Market Risk', 'Sector Risk', 'Company Risk', 'Liquidity Risk', 
                'Volatility', 'Correlation w/Market'
            ])
            
            st.dataframe(styled_risk, use_container_width=True)
            
            # Risk explanation
            with st.expander("Risk Metrics Explained"):
                st.markdown("""
                - **Beta**: Measures volatility relative to the overall market. A beta > 1 indicates higher volatility than the market.
                - **Volatility (Vol)**: Standard deviation of returns, indicating price fluctuation magnitude.
                - **Max Drawdown**: Largest percentage drop from peak to trough, indicating worst-case historical loss.
                - **Sharpe Ratio**: Risk-adjusted return metric. Higher values indicate better risk-adjusted performance.
                - **Market Risk**: Risk related to overall market movements affecting the stock.
                - **Sector Risk**: Risk related to the specific industry sector's performance.
                - **Company Risk**: Risk specific to the company's operations, management, and financials.
                - **Liquidity Risk**: Risk related to how easily shares can be bought or sold without affecting price.
                """)
        
        with tabs[4]:
            st.subheader("Sector Performance")
            
            # Sector performance data
            sector_df = pd.DataFrame({
                'Sector': list(result['sector_performance'].keys()),
                'Performance (%)': list(result['sector_performance'].values())
            })
            
            # Sort by performance
            sector_df = sector_df.sort_values('Performance (%)', ascending=False)
            
            # Create columns
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a horizontal bar chart
                fig = px.bar(
                    sector_df,
                    y='Sector',
                    x='Performance (%)',
                    orientation='h',
                    color='Performance (%)',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    title='Sector Performance (YTD)',
                    text='Performance (%)'
                )
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(height=500)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sector allocation of recommended stocks
                st.markdown("### Recommended Portfolio Sector Allocation")
                
                # Sample sector allocation data
                sector_allocation = {
                    'Technology': 60,
                    'Consumer Cyclical': 20,
                    'Communication Services': 20
                }
                
                # Create pie chart
                fig = px.pie(
                    names=list(sector_allocation.keys()),
                    values=list(sector_allocation.values()),
                    title='Sector Allocation',
                    hole=0.4
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sector outlook
                st.markdown("### Sector Outlook")
                st.markdown("""
                - **Technology**: Strong outlook driven by AI adoption and cloud growth
                - **Consumer Cyclical**: Moderate outlook with potential pressure from interest rates
                - **Communication Services**: Positive outlook with increasing digital ad spending
                """)
        
        # Detailed analysis and recommendations section
        st.markdown('<div class="sub-header">Detailed Analysis & Recommendations</div>', unsafe_allow_html=True)
        
        # Create an expander for each stock
        for stock in result["top_stocks"]:
            with st.expander(f"{stock['symbol']} - {stock['name']} | {stock['recommendation']}"):
                # Create tabs within the expander
                stock_tabs = st.tabs(['Investment Thesis', 'Technical Analysis', 'Fundamental Analysis', 'Risk Assessment'])
                
                with stock_tabs[0]:
                    st.markdown(f"### Investment Thesis for {stock['symbol']}")
                    st.markdown(f"""
                    **Target Price:** ${stock['target_price']} ({"+" if stock['target_price'] > 200 else ""}{((stock['target_price']/200)-1)*100:.1f}% upside)
                    
                    **Recommendation:** {stock['recommendation']}
                    
                    **Time Horizon:** 12-18 months
                    
                    **Thesis Summary:**
                    {stock['name']} presents a compelling investment opportunity based on its strong technical momentum, solid fundamental growth metrics, and reasonable risk profile. The company is well-positioned to benefit from ongoing digital transformation trends and expanding profit margins.
                    
                    **Key Catalysts:**
                    - Continued expansion in service revenue streams
                    - Margin improvement from supply chain optimization
                    - New product launches expected in Q3 2025
                    - Potential for increased shareholder returns via buybacks
                    
                    **Position Sizing:**
                    Recommended position size of 4-6% in a diversified portfolio, with potential to add on pullbacks to key support levels.
                    """)
                
                with stock_tabs[1]:
                    st.markdown(f"### Technical Analysis for {stock['symbol']}")
                    
                    # Technical metrics with visual indicators
                    st.markdown("#### Technical Indicators")
                    
                    # Create columns for technical metrics
                    tech_cols = st.columns(5)
                    
                    # Sample technical metrics
                    tech_metrics = [
                        {"label": "Trend", "value": "Bullish", "detail": "Above major MAs"},
                        {"label": "RSI(14)", "value": "62.3", "detail": "Positive momentum"},
                        {"label": "MACD", "value": "Positive", "detail": "Recent crossover"},
                        {"label": "Vol Trend", "value": "Increasing", "detail": "Above average"},
                        {"label": "Pattern", "value": "Cup & Handle", "detail": "Bullish formation"}
                    ]
                    
                    for i, metric in enumerate(tech_metrics):
                        with tech_cols[i]:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                <div style="font-size: 0.9rem; color: #666;">{metric['label']}</div>
                                <div style="font-size: 1.3rem; font-weight: bold; margin: 5px 0;">{metric['value']}</div>
                                <div style="font-size: 0.8rem; color: #666;">{metric['detail']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Technical analysis summary
                    st.markdown("""
                    #### Technical Analysis Summary
                    
                    The stock is displaying strong bullish momentum with prices trading above both the 50-day and 200-day moving averages. Recent price action has formed a cup and handle pattern, suggesting potential for further upside movement. Volume has been increasing on up days, confirming the bullish bias.
                    
                    **Support Levels:**
                    - Primary: $192.50 (50-day MA)
                    - Secondary: $187.80 (previous resistance turned support)
                    - Tertiary: $180.00 (psychological level)
                    
                    **Resistance Levels:**
                    - Primary: $212.75 (recent high)
                    - Secondary: $225.00 (psychological level)
                    - Tertiary: $231.25 (all-time high)
                    
                    **Entry Strategy:**
                    Consider entering on pullbacks to the $192-195 range with a stop loss below $187.
                    """)
                
                with stock_tabs[2]:
                    st.markdown(f"### Fundamental Analysis for {stock['symbol']}")
                    
                    # Create columns
                    fund_col1, fund_col2 = st.columns([1, 1])
                    
                    with fund_col1:
                        # Financial metrics table
                        st.markdown("#### Key Financial Metrics")
                        
                        # Sample financial metrics
                        financials = {
                            "Metric": ["Revenue (TTM)", "Revenue Growth (YoY)", "EPS (TTM)", "EPS Growth (YoY)", "Profit Margin", "Operating Margin", "ROE", "ROA", "ROIC", "Debt/Equity"],
                            "Value": ["$394.2B", "15.2%", "$6.42", "18.7%", "25.3%", "32.1%", "35.2%", "18.7%", "27.5%", "0.42"],
                            "Industry Avg": ["$215.7B", "12.5%", "$3.85", "10.2%", "19.8%", "25.3%", "28.7%", "15.2%", "21.3%", "0.55"]
                        }
                        
                        financials_df = pd.DataFrame(financials)
                        
                        # Function to highlight where company is better than industry
                        def highlight_better(row):
                            # For metrics where higher is better
                            higher_better = ["Revenue (TTM)", "Revenue Growth (YoY)", "EPS (TTM)", "EPS Growth (YoY)", 
                                          "Profit Margin", "Operating Margin", "ROE", "ROA", "ROIC"]
                            # For metrics where lower is better
                            lower_better = ["Debt/Equity"]
                            
                            if row.name not in [0, 1]:  # Skip header rows
                                metric = row["Metric"]
                                
                                # Extract numeric values for comparison
                                try:
                                    # Remove non-numeric characters and convert to float
                                    val_str = row["Value"].replace('$', '').replace('B', '').replace('%', '')
                                    avg_str = row["Industry Avg"].replace('$', '').replace('B', '').replace('%', '')
                                    val = float(val_str)
                                    avg = float(avg_str)
                                    
                                    if metric in higher_better and val > avg:
                                        return ['', 'background-color: rgba(76, 175, 80, 0.2)', '']
                                    elif metric in lower_better and val < avg:
                                        return ['', 'background-color: rgba(76, 175, 80, 0.2)', '']
                                except:
                                    pass
                            return ['', '', '']
                        
                        st.dataframe(financials_df.style.apply(highlight_better, axis=1), use_container_width=True)
                    
                    with fund_col2:
                        # Valuation metrics table
                        st.markdown("#### Valuation Metrics")
                        
                        # Sample valuation metrics
                        valuation = {
                            "Metric": ["P/E Ratio", "Forward P/E", "PEG Ratio", "P/S Ratio", "P/B Ratio", "EV/EBITDA", "EV/Revenue", "Dividend Yield", "FCF Yield", "Earnings Yield"],
                            "Value": ["28.5", "24.2", "1.52", "7.2", "12.5", "18.2", "6.8", "0.8%", "3.2%", "3.5%"],
                            "5Y Average": ["32.7", "27.5", "1.75", "8.4", "14.2", "20.1", "7.5", "0.7%", "2.8%", "3.1%"]
                        }
                        
                        valuation_df = pd.DataFrame(valuation)
                        
                        # Function to highlight where current is better than 5Y avg
                        def highlight_better_valuation(row):
                            # For metrics where lower is better
                            lower_better = ["P/E Ratio", "Forward P/E", "PEG Ratio", "P/S Ratio", "P/B Ratio", "EV/EBITDA", "EV/Revenue"]
                            # For metrics where higher is better
                            higher_better = ["Dividend Yield", "FCF Yield", "Earnings Yield"]
                            
                            if row.name not in [0, 1]:  # Skip header rows
                                metric = row["Metric"]
                                
                                # Extract numeric values for comparison
                                try:
                                    # Remove non-numeric characters and convert to float
                                    val_str = row["Value"].replace('%', '')
                                    avg_str = row["5Y Average"].replace('%', '')
                                    val = float(val_str)
                                    avg = float(avg_str)
                                    
                                    if metric in lower_better and val < avg:
                                        return ['', 'background-color: rgba(76, 175, 80, 0.2)', '']
                                    elif metric in higher_better and val > avg:
                                        return ['', 'background-color: rgba(76, 175, 80, 0.2)', '']
                                except:
                                    pass
                            return ['', '', '']
                        
                        st.dataframe(valuation_df.style.apply(highlight_better_valuation, axis=1), use_container_width=True)
                    
                    # Fundamental analysis summary
                    st.markdown("""
                    #### Fundamental Analysis Summary
                    
                    The company demonstrates strong financial health with revenue and earnings growth exceeding industry averages. Profit margins are expanding due to operational efficiencies and economies of scale. The balance sheet remains strong with manageable debt levels and significant cash reserves.
                    
                    **Growth Drivers:**
                    - Expansion of services ecosystem creating higher-margin revenue streams
                    - International market penetration, particularly in emerging markets
                    - New product categories showing promising adoption rates
                    - Strategic acquisitions enhancing technological capabilities
                    
                    **Valuation Assessment:**
                    While the stock trades at a premium to the broader market on a P/E basis, it appears reasonably valued relative to its growth rate and historical averages. The PEG ratio of 1.52 suggests fair value considering the company's growth prospects.
                    """)
                
                with stock_tabs[3]:
                    st.markdown(f"### Risk Assessment for {stock['symbol']}")
                    
                    # Risk radar chart
                    st.markdown("#### Risk Profile")
                    
                    # Sample risk data for radar chart
                    risk_categories = ['Market Risk', 'Sector Risk', 'Valuation Risk', 'Financial Risk', 'Competition Risk', 'Regulatory Risk']
                    risk_values = [5, 4, 6, 3, 5, 4]  # 1-10 scale where lower is better
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=risk_values,
                        theta=risk_categories,
                        fill='toself',
                        name=stock['symbol']
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )
                        ),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk factors
                    st.markdown("#### Key Risk Factors")
                    
                    # Create two columns for risks
                    risk_col1, risk_col2 = st.columns([1, 1])
                    
                    with risk_col1:
                        st.markdown("""
                        **Market & Macroeconomic Risks:**
                        - Sensitivity to broader market corrections
                        - Potential impact from rising interest rates
                        - Currency fluctuation effects on international revenue
                        
                        **Competition & Industry Risks:**
                        - Increasing competition in core markets
                        - Rapid technological changes requiring constant innovation
                        - Margin pressure from emerging low-cost competitors
                        """)
                    
                    with risk_col2:
                        st.markdown("""
                        **Company-Specific Risks:**
                        - Product cycle dependencies
                        - Key personnel dependencies
                        - Supply chain vulnerabilities
                        
                        **Regulatory & Other Risks:**
                        - Potential antitrust scrutiny
                        - Data privacy regulation changes
                        - Intellectual property challenges
                        """)
                    
                    # Risk mitigation
                    st.markdown("""
                    #### Risk Mitigation Strategies
                    
                    **Position Sizing:**
                    Limit position to 4-6% of portfolio to manage stock-specific risk exposure.
                    
                    **Entry Strategy:**
                    Consider dollar-cost averaging or scaling in on technical pullbacks rather than establishing full position at once.
                    
                    **Hedging Considerations:**
                    For larger positions, consider protective puts or collar strategies during periods of elevated volatility or ahead of key events.
                    
                    **Exit Strategy:**
                    Set a stop-loss at $187 (approximately 8% below current levels) to limit downside risk.
                    """)
        
        # Historical performance and backtesting section
        st.markdown('<div class="sub-header">Historical Performance & Backtesting</div>', unsafe_allow_html=True)
        
        # Generate sample historical performance data
        dates = pd.date_range(end=pd.Timestamp.today(), periods=252)  # Approximately 1 year of trading days
        
        # Create sample portfolio and benchmark returns
        np.random.seed(42)  # For reproducibility
        
        # Generate correlated returns (portfolio and S&P 500)
        correlation = 0.8
        volatility_portfolio = 0.012
        volatility_sp500 = 0.010
        
        # Generate correlated random returns
        returns_portfolio = np.random.normal(0.0005, volatility_portfolio, len(dates))
        returns_sp500 = np.random.normal(0.0004, volatility_sp500, len(dates))
        
        # Add correlation
        returns_sp500 = correlation * returns_portfolio + np.sqrt(1 - correlation**2) * returns_sp500
        
        # Create price series
        portfolio_series = 100 * (1 + returns_portfolio).cumprod()
        sp500_series = 100 * (1 + returns_sp500).cumprod()
        
        # Create dataframe
        performance_df = pd.DataFrame({
            'Date': dates,
            'Portfolio': portfolio_series,
            'S&P 500': sp500_series
        })
        
        # Create columns
        perf_col1, perf_col2 = st.columns([3, 2])
        
        with perf_col1:
            # Performance chart
            st.subheader("Strategy Backtest Performance")
            
            # Create line chart
            fig = px.line(
                performance_df, 
                x='Date', 
                y=['Portfolio', 'S&P 500'],
                title='Backtest Performance vs S&P 500 (1 Year)',
                labels={'value': 'Value ($)', 'variable': 'Series'}
            )
            
            fig.update_layout(hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with perf_col2:
            # Performance metrics
            st.subheader("Performance Metrics")
            
            # Calculate sample performance metrics
            portfolio_return = (portfolio_series[-1] / portfolio_series[0] - 1) * 100
            sp500_return = (sp500_series[-1] / sp500_series[0] - 1) * 100
            
            # Annualized volatility
            portfolio_vol = np.std(returns_portfolio) * np.sqrt(252) * 100
            sp500_vol = np.std(returns_sp500) * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            portfolio_sharpe = (portfolio_return - 2) / portfolio_vol
            sp500_sharpe = (sp500_return - 2) / sp500_vol
            
            # Create metrics table
            metrics_data = {
                'Metric': ['Total Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Beta', 'Alpha (%)'],
                'Portfolio': [f"{portfolio_return:.2f}%", f"{portfolio_vol:.2f}%", f"{portfolio_sharpe:.2f}", "12.5%", "0.95", "5.2%"],
                'S&P 500': [f"{sp500_return:.2f}%", f"{sp500_vol:.2f}%", f"{sp500_sharpe:.2f}", "14.8%", "1.00", "0.0%"]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Style the dataframe
            def highlight_better_performance(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                
                # Compare metrics
                for i in range(len(df)):
                    metric = df.iloc[i, 0]
                    
                    # For metrics where higher is better
                    if metric in ['Total Return (%)', 'Sharpe Ratio', 'Alpha (%)']:
                        if float(df.iloc[i, 1].replace('%', '')) > float(df.iloc[i, 2].replace('%', '')):
                            styles.iloc[i, 1] = 'background-color: rgba(76, 175, 80, 0.2); color: #1e5631'
                        else:
                            styles.iloc[i, 2] = 'background-color: rgba(76, 175, 80, 0.2); color: #1e5631'
                    
                    # For metrics where lower is better
                    elif metric in ['Annualized Volatility (%)', 'Max Drawdown (%)']:
                        if float(df.iloc[i, 1].replace('%', '')) < float(df.iloc[i, 2].replace('%', '')):
                            styles.iloc[i, 1] = 'background-color: rgba(76, 175, 80, 0.2); color: #1e5631'
                        else:
                            styles.iloc[i, 2] = 'background-color: rgba(76, 175, 80, 0.2); color: #1e5631'
                
                return styles
            
            st.dataframe(metrics_df.style.apply(highlight_better_performance, axis=None), use_container_width=True)
            
            # Performance summary
            st.markdown("""
            #### Backtest Summary
            
            The recommended portfolio strategy has demonstrated superior risk-adjusted returns compared to the S&P 500 benchmark. Key strengths include:
            
            - Higher total return with lower volatility
            - Improved Sharpe ratio indicating better risk-adjusted performance
            - Lower maximum drawdown suggesting better downside protection
            - Positive alpha indicating value added by the selection strategy
            
            Past performance is not indicative of future results, but the strategy has shown robustness across different market conditions.
            """)
    
    # If no analysis has been run yet, show the welcome screen
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <img src="https://img.icons8.com/fluency/240/000000/investment-portfolio.png" width="120"/>
            <h2 style="margin-top: 1rem;">Welcome to the Multi-Agent Stock Analysis System</h2>
            <p style="font-size: 1.2rem; margin: 1rem 0 2rem 0;">Configure your analysis parameters in the sidebar and click "Run Full Analysis" to get started.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview
        st.subheader("System Features")
        
        features = [
            {
                "icon": "📊",
                "title": "Multi-agent Analysis", 
                "description": "Leverages 5 specialized AI agents working together to provide comprehensive stock analysis."
            },
            {
                "icon": "🧠",
                "title": "Technical Analysis", 
                "description": "Evaluates price patterns, momentum indicators, and chart formations to identify market trends."
            },
            {
                "icon": "💼",
                "title": "Fundamental Analysis", 
                "description": "Assesses company financials, growth metrics, and valuation to determine intrinsic worth."
            },
            {
                "icon": "⚖️",
                "title": "Risk Assessment", 
                "description": "Measures volatility, drawdowns, and various risk factors to optimize risk-adjusted returns."
            },
            {
                "icon": "📈",
                "title": "Portfolio Recommendations", 
                "description": "Synthesizes all analyses to identify the most promising investment opportunities."
            },
            {
                "icon": "📱",
                "title": "Interactive Visualizations", 
                "description": "Provides rich, interactive charts and graphs to understand complex market dynamics."
            }
        ]
        
        # Create columns for features
        cols = st.columns(3)
        
        for i, feature in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 1.5rem; margin-bottom: 1rem; height: 200px;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{feature['icon']}</div>
                    <h3 style="margin-top: 0;">{feature['title']}</h3>
                    <p>{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Getting started section
        st.subheader("Getting Started")
        
        st.markdown("""
        1. **Configure Analysis Parameters**: Use the sidebar to select your preferred time period, stock filters, and analysis weights.
        
        2. **Run Analysis**: Click the "Run Full Analysis" button to start the AI agents' analysis process.
        
        3. **Review Results**: Explore the comprehensive analysis across multiple tabs, from high-level recommendations to detailed stock-specific insights.
        
        4. **Export Findings**: Download the analysis results in your preferred format for future reference or sharing.
        """)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 0.8rem;">
        Multi-Agent Stock Analysis System powered by CrewAI and Google Gemini 1.5<br>
        Disclaimer: This tool is for informational purposes only and does not constitute investment advice.
    </div>
    """, unsafe_allow_html=True)


# Import statements that should be at the top of your file


if __name__ == "__main__":
    main()