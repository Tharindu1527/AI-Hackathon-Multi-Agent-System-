from crewai import Agent

def create_data_collection_agent(llm):
    """Create the Data Collection Agent"""
    return Agent(
        role="Data Collection Specialist",
        goal="Gather comprehensive stock data from multiple financial APIs",
        backstory="""You are a data specialist with extensive experience in financial markets.
        Your expertise lies in collecting and organizing data from various financial sources.
        You know how to query APIs efficiently and structure data for further analysis.""",
        verbose=True,
        allow_delegation=True,
        llm=llm
    )