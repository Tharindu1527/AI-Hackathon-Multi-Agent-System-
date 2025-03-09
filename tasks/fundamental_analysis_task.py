from crewai import Task

def create_fundamental_analysis_task(agent, context_tasks):
    """Create the Fundamental Analysis Task"""
    return Task(
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
        agent=agent,
        expected_output="""A comprehensive fundamental analysis for each stock, including profitability, 
        valuation, growth prospects, balance sheet analysis, and a final ranking of the top 10 stocks based 
        on fundamental strength with clear justification.""",
        context=context_tasks
    )