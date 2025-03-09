from crewai import Task

def create_portfolio_recommendation_task(agent, context_tasks):
    """Create the Portfolio Recommendation Task"""
    return Task(
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
        agent=agent,
        expected_output="""A professional investment recommendation report identifying the top 5 investable 
        stocks with comprehensive justification, supported by technical, fundamental, and risk analyses. 
        Include specific entry points, price targets, and holding periods.""",
        context=context_tasks
    )