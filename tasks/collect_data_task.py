from crewai import Task

def create_collect_data_task(agent):
    """Create the Data Collection Task"""
    return Task(
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
        agent=agent,
        expected_output="""A detailed JSON containing structured data from all three APIs for at least 30 major stocks, 
        ready for analysis by other agents. Ensure it includes historical prices, key financial metrics, 
        and fundamental information."""
    )