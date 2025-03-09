from crewai import Agent

def create_technical_analysis_agent(llm):
    """Create the Technical Analysis Agent"""
    return Agent(
        role="Technical Analysis Expert",
        goal="Perform in-depth technical analysis on stock data to identify patterns and trends",
        backstory="""You are a seasoned technical analyst with years of experience in chart patterns, 
        technical indicators, and price action analysis. You can spot trends and reversals that 
        others might miss. Your analysis is rooted in statistical evidence and historical patterns.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )