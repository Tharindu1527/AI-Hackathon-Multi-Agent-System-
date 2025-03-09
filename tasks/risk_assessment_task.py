from crewai import Task

def create_risk_assessment_task(agent, context_tasks):
    """Create the Risk Assessment Task"""
    return Task(
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
        agent=agent,
        expected_output="""A detailed risk assessment for each stock, including volatility metrics, 
        drawdown analysis, liquidity assessment, key risk factors, and a final ranking of the top 10 stocks 
        with the most favorable risk-reward profiles.""",
        context=context_tasks
    )