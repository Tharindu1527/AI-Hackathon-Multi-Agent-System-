from crewai import Task

def create_technical_analysis_task(agent, context_tasks):
    """Create the Technical Analysis Task"""
    return Task(
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
        agent=agent,
        expected_output="""A detailed technical analysis report for each stock, including trend analysis, 
        indicator readings, support/resistance levels, and a final ranking of the top 10 stocks based on 
        technical strength with clear justification.""",
        context=context_tasks
    )