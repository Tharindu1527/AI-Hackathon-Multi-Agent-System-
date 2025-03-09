from datetime import datetime

def format_analysis_results(raw_result):
    """Format the raw results from the CrewAI agents into a structured output"""
    
    # Initialize the formatted result structure
    formatted_result = {
        "top_stocks": [],
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_outlook": "",
        "sector_performance": {}
    }
    
    # Access the task outputs directly from the CrewOutput object
    tasks_output = raw_result.tasks_output
    
    # Initialize variables to store task results
    technical_analysis = {}
    fundamental_analysis = {}
    risk_assessment = {}
    portfolio_recommendation = {}
    
    # Extract data from the task outputs
    for task_output in tasks_output:
        if isinstance(task_output.raw, dict):
            if "technical_analysis" in task_output.raw:
                technical_analysis = task_output.raw["technical_analysis"]
            elif "fundamental_analysis" in task_output.raw:
                fundamental_analysis = task_output.raw["fundamental_analysis"]
            elif "risk_assessment" in task_output.raw:
                risk_assessment = task_output.raw["risk_assessment"]
            elif "portfolio_recommendation" in task_output.raw:
                portfolio_recommendation = task_output.raw["portfolio_recommendation"]
    
    # Process the portfolio recommendation data
    if portfolio_recommendation:
        formatted_result["market_outlook"] = portfolio_recommendation.get("market_outlook", 
                                           "Market outlook not provided by analysis.")
        formatted_result["sector_performance"] = portfolio_recommendation.get("sector_performance", {})
        
        recommended_stocks = portfolio_recommendation.get("recommended_stocks", [])
        for stock in recommended_stocks[:5]:
            stock_data = {
                "symbol": stock.get("symbol"),
                "name": stock.get("name"),
                "technical_score": float(stock.get("technical_score", 0)),
                "fundamental_score": float(stock.get("fundamental_score", 0)),
                "risk_score": float(stock.get("risk_score", 0)),
                "composite_score": float(stock.get("composite_score", 0)),
                "recommendation": stock.get("recommendation"),
                "target_price": float(stock.get("target_price", 0))
            }
            formatted_result["top_stocks"].append(stock_data)
    
    return formatted_result