from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
import requests
from datetime import datetime

load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.environ.get('SERPAPI_API_KEY')

# For normal accurate responses
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

# Load SerpAPI tool
toolbox = load_tools(tool_names=['serpapi'], llm=llm)

@tool
def get_location_from_ip():
    """Get the geographical location based on the IP address."""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            data = (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {data.get('city', 'N/A')},\n"
                f"Country: {data.get('country', 'N/A')}"
            )
            return data
        else:
            return "Location could not be determined."
    except Exception as e:
        return f"Error occurred: {e}"

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = toolbox + [get_current_datetime, get_location_from_ip]

agent = create_tool_calling_agent(
    llm=llm, tools=tools, prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True
)

# Example usage
if __name__ == "__main__":
    # Test the agent with a weather query
    result = agent_executor.invoke({"input": "What's the weather here?"})
    print(result)





#     # Required imports
# from langchain_community.agent_toolkits import load_tools
# from langchain_openai import ChatOpenAI
# from langchain.agents import create_tool_calling_agent
# from langchain.agents import AgentExecutor
# from langchain_core.tools import tool

# # Initialize LLM
# llm = ChatOpenAI(temperature=0)

# # Complete list of available tools
# tools = load_tools([
#     # Search and Information Tools
#     "serpapi",        # Google search results via SerpAPI
#     "google-search",  # Direct Google search
#     "wikipedia",      # Wikipedia article search
#     "duckduckgo",     # DuckDuckGo search
#     "arxiv",          # ArXiv paper search
#     "pubmed",         # PubMed medical paper search
#     "newsapi",        # News article search
#     "tmdb",           # The Movie Database search
    
#     # Math and Calculation Tools
#     "llm-math",       # Mathematical calculations using LLM
#     "calculator",     # Basic calculator operations
    
#     # Web and API Tools
#     "requests",       # Make HTTP requests
#     "requests_all",   # Make HTTP requests with more options
#     "http",          # HTTP client
#     "json",          # JSON operations
#     "openweathermap", # Weather data
#     "youtube",        # YouTube search
#     "reddit",         # Reddit search
    
#     # File System Tools
#     "file_system",    # File operations
#     "csv",           # CSV file operations
#     "json_file",     # JSON file operations
#     "text_file",     # Text file operations
    
#     # Database Tools
#     "sql_database",   # SQL database operations
#     "mongodb",        # MongoDB operations
#     "redis",          # Redis operations
    
#     # Document Processing Tools
#     "pdf",           # PDF processing
#     "docx",          # Word document processing
#     "txt",           # Text file processing
#     "markdown",      # Markdown processing
    
#     # Code and Development Tools
#     "python_repl",    # Python REPL
#     "shell",          # Shell commands
#     "git",            # Git operations
    
#     # Data Analysis Tools
#     "pandas",         # Pandas DataFrame operations
#     "numpy",          # NumPy operations
#     "matplotlib",     # Matplotlib plotting
    
#     # Language and Text Tools
#     "translator",     # Language translation
#     "summarizer",     # Text summarization
#     "sentiment",      # Sentiment analysis
# ], llm=llm)

# # Required pip installations
# """
# pip install google-search-results  # For SerpAPI
# pip install wikipedia             # For Wikipedia
# pip install requests              # For HTTP requests
# pip install python-dotenv         # For environment variables
# pip install pandas               # For data analysis
# pip install numpy                # For numerical operations
# pip install matplotlib           # For plotting
# pip install pymongo              # For MongoDB
# pip install redis                # For Redis
# pip install python-docx          # For Word documents
# pip install PyPDF2               # For PDF processing
# pip install newsapi-python       # For News API
# pip install arxiv                # For ArXiv
# pip install pubmed-lookup        # For PubMed
# pip install tmdbv3api            # For TMDB
# pip install youtube-search       # For YouTube
# pip install praw                 # For Reddit
# pip install openweathermap-api   # For OpenWeatherMap
# pip install python-git           # For Git operations
# """

# # Example of creating a custom tool
# @tool
# def custom_tool():
#     """Description of what the tool does."""
#     # Tool implementation
#     pass

# # Create agent with tools
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=tools,
#     prompt=prompt
# )

# # Create executor
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True
# )

# # Example usage
# if __name__ == "__main__":
#     # Test the agent with a query
#     result = agent_executor.invoke({"input": "What's the weather in New York?"})
#     print(result)