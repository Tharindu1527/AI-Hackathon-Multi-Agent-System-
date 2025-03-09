from tasks.collect_data_task import create_collect_data_task
from tasks.technical_analysis_task import create_technical_analysis_task
from tasks.fundamental_analysis_task import create_fundamental_analysis_task
from tasks.risk_assessment_task import create_risk_assessment_task
from tasks.portfolio_recommendation_task import create_portfolio_recommendation_task

def create_all_tasks(agents):
    data_collection_agent = agents[0]
    technical_analysis_agent = agents[1]
    fundamental_analysis_agent = agents[2]
    risk_assessment_agent = agents[3]
    portfolio_recommendation_agent = agents[4]
    
    # Create tasks in sequence to establish dependencies
    collect_data_task = create_collect_data_task(data_collection_agent)
    
    technical_analysis_task = create_technical_analysis_task(
        technical_analysis_agent, 
        [collect_data_task]
    )
    
    fundamental_analysis_task = create_fundamental_analysis_task(
        fundamental_analysis_agent, 
        [collect_data_task]
    )
    
    risk_assessment_task = create_risk_assessment_task(
        risk_assessment_agent, 
        [collect_data_task, technical_analysis_task, fundamental_analysis_task]
    )
    
    portfolio_recommendation_task = create_portfolio_recommendation_task(
        portfolio_recommendation_agent, 
        [technical_analysis_task, fundamental_analysis_task, risk_assessment_task]
    )
    
    return [
        collect_data_task,
        technical_analysis_task,
        fundamental_analysis_task,
        risk_assessment_task,
        portfolio_recommendation_task
    ]