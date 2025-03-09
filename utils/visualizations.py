import streamlit as st
import plotly.graph_objects as go

def display_market_outlook(market_outlook):
    """Display the market outlook section"""
    st.header("Market Outlook")
    st.write(market_outlook)

def display_sector_performance(sector_data):
    """Display sector performance chart"""
    st.header("Sector Performance")
    
    if not sector_data:
        st.warning("No sector performance data available.")
        return
        
    fig = go.Figure(data=[go.Bar(x=list(sector_data.keys()), y=list(sector_data.values()))])
    fig.update_layout(title="Sector Performance", xaxis_title="Sector", yaxis_title="Performance")
    st.plotly_chart(fig)

def display_top_stocks(top_stocks):
    """Display top recommended stocks"""
    st.header("Top 5 Recommended Stocks")
    for stock in top_stocks:
        with st.expander(f"{stock['name']} ({stock['symbol']})"):
            st.markdown(f"""
            **Technical Score:** {stock['technical_score']:.2f}  
            **Fundamental Score:** {stock['fundamental_score']:.2f}  
            **Risk Score:** {stock['risk_score']:.2f}  
            **Composite Score:** {stock['composite_score']:.2f}  
            **Recommendation:** {stock['recommendation']}  
            **Target Price:** ${stock['target_price']:.2f}
            """)

def display_stocks_comparison(top_stocks):
    """Display a comparison chart for the top stocks"""
    fig = go.Figure()
    for stock in top_stocks:
        fig.add_trace(go.Bar(
            x=['Technical', 'Fundamental', 'Risk', 'Composite'],
            y=[stock['technical_score'], stock['fundamental_score'], stock['risk_score'], stock['composite_score']],
            name=stock['symbol']
        ))
    fig.update_layout(title="Top Stocks Comparison", barmode='group')
    st.plotly_chart(fig)