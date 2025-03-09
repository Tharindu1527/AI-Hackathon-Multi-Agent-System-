from crewai import LLM
from agents.data_collection_agent import create_data_collection_agent
from agents.technical_analysis_agent import create_technical_analysis_agent
from agents.fundamental_analysis_agent import create_fundamental_analysis_agent
from agents.risk_assessment_agent import create_risk_assessment_agent
from agents.portfolio_recommendation_agent import create_portfolio_recommendation_agent

# Initialize LLM
def get_llm():
    return LLM(
        model="gemini/gemini-1.5-flash",
        temperature=0.7
    )

def create_all_agents():
    llm = get_llm()
    
    return [
        create_data_collection_agent(llm),
        create_technical_analysis_agent(llm),
        create_fundamental_analysis_agent(llm),
        create_risk_assessment_agent(llm),
        create_portfolio_recommendation_agent(llm)
    ]