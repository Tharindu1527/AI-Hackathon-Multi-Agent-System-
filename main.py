import streamlit as st
import asyncio
import nest_asyncio
from crewai import Crew, Process
from agents import create_all_agents
from tasks import create_all_tasks
from utils.data_processors import format_analysis_results
from utils.visualizations import (
    display_market_outlook, 
    display_sector_performance,
    display_top_stocks,
    display_stocks_comparison
)

# Apply nest_asyncio to fix asyncio event loop issues
nest_asyncio.apply()

def run_stock_analysis():
    """Execute the CrewAI workflow and return formatted results"""
    try:
        # Create agents and tasks
        agents = create_all_agents()
        tasks = create_all_tasks(agents)
        
        # Create the Crew
        stock_analysis_crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        # Execute the crew workflow
        raw_result = stock_analysis_crew.kickoff()
        
        # Format the results
        return format_analysis_results(raw_result)
        
    except Exception as e:
        print(f"Error in stock analysis: {str(e)}")
        raise e

def main():
    st.title("AI-Powered Stock Analysis")

    # Add a button to trigger the analysis
    if st.button("Run Stock Analysis"):
        with st.spinner("Analyzing stocks... This may take a few minutes."):
            # Fetch stock data from AI agents
            analysis_results = run_stock_analysis()

        if analysis_results:
            # Display market outlook
            display_market_outlook(analysis_results["market_outlook"])
            
            # Display sector performance
            display_sector_performance(analysis_results["sector_performance"])
            
            # Display top stocks
            display_top_stocks(analysis_results["top_stocks"])
            
            # Create a comparison chart for the top stocks
            display_stocks_comparison(analysis_results["top_stocks"])
        else:
            st.warning("No stock data available. Ensure agents are responding.")

    st.sidebar.markdown("## About")
    st.sidebar.info("This AI-powered stock analysis tool uses advanced agents to analyze market data and provide investment recommendations.")

if __name__ == "__main__":
    main()