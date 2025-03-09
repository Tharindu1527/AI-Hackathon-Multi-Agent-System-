import os
from datetime import datetime
from dotenv import load_dotenv
from langfuse.client import Langfuse

# Load environment variables
load_dotenv()

# API Keys
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Langfuse for telemetry
langfuse = Langfuse(
    secret_key=LANGFUSE_SECRET_KEY,
    public_key=LANGFUSE_PUBLIC_KEY,
)

# Create a trace for the entire process
trace = langfuse.trace(
    name="Stock Analysis System",
    metadata={"timestamp": datetime.now().isoformat()}
)

# List of default stocks to analyze
DEFAULT_STOCKS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]