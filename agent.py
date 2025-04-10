import os
from typing import Dict, List, TypedDict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
import requests
from datetime import datetime
import json
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define types for better type checking
class NewsArticle(TypedDict):
    title: str
    source: str
    url: str
    published_at: str
    summary: str


class AgentState(TypedDict, total=False):
    """State for the multi-agent blog system"""
    messages: List[Dict[str, str]]
    news_data: List[NewsArticle]
    draft_blog_post: str
    proofread_feedback: str
    final_blog_post: str


# Tool for searching Washington Nationals news
@tool
def search_nationals_news(_: None = None) -> List[NewsArticle]:
    """
    Search for the latest Washington Nationals game news.
    Returns recent articles about Nationals games.
    """
    # Use Google Custom Search API with credentials from .env file
    # Make sure .env file contains GOOGLE_API_KEY and GOOGLE_CSE_ID

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    # Get credentials from environment variables loaded from .env
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_cse_id = os.environ.get("GOOGLE_CSE_ID")
    
    if not google_api_key or not google_cse_id:
        print("Warning: Google API credentials not found in .env file. Using mock data.")
        return get_mock_nationals_news()
        
    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": "Washington Nationals game most recent game news from last 24 hours",
        "num": 5
    }

    try:
        response = requests.get(search_url, params=params, headers=headers)
        if response.status_code == 200:
            results = response.json().get("items", [])
            articles = []

            print("\n=== SEARCH RESULTS ===")
            for i, item in enumerate(results):
                title = item.get("title", "")
                url = item.get("link", "")
                print(f"Found article {i+1}: {title} at {url}")
                
                articles.append({
                    "title": title,
                    "source": item.get("displayLink", ""),
                    "url": url,
                    "published_at": datetime.now().isoformat(),
                    "summary": item.get("snippet", "")
                })

            return articles
        else:
            # Return mock data if the search fails
            print(f"Error: {response.status_code} - {response.text}")
            return get_mock_nationals_news()
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        return get_mock_nationals_news()


def get_mock_nationals_news() -> List[NewsArticle]:
    """Return mock data for demonstration purposes"""
    print("Using mock data for Washington Nationals news")
    return [
        {
            "title": "Nationals defeat Marlins 5-3 behind CJ Abrams' home run",
            "source": "washingtonpost.com",
            "url": "https://www.washingtonpost.com/sports/nationals",
            "published_at": datetime.now().isoformat(),
            "summary": "CJ Abrams hit a two-run homer and the Washington Nationals defeated the Miami Marlins 5-3 on Sunday."
        },
        {
            "title": "Nationals' MacKenzie Gore strikes out 10 in win over Phillies",
            "source": "mlb.com",
            "url": "https://www.mlb.com/nationals/news",
            "published_at": datetime.now().isoformat(),
            "summary": "Left-hander MacKenzie Gore struck out 10 batters over six innings as the Nationals beat the Phillies 4-2."
        },
        {
            "title": "James Wood makes spectacular catch in Nationals victory",
            "source": "nbcsports.com",
            "url": "https://www.nbcsports.com/washington/nationals",
            "published_at": datetime.now().isoformat(),
            "summary": "Rookie outfielder James Wood made a diving catch to save two runs in the Nationals' 3-1 win over the Braves."
        }
    ]


# Define the agents (nodes in the graph)
def news_agent(state) -> AgentState:
    """
    Agent responsible for searching news
    """
    print("Running news_agent...")
    
    # Create a copy of the state to avoid modifying the input
    new_state = state.copy()

    # Search for news if we don't have it yet
    if "news_data" not in state or not state["news_data"]:
        # Get the news
        news_data = search_nationals_news.invoke(None)
        new_state["news_data"] = news_data

        # Add a log message
        if "messages" not in new_state:
            new_state["messages"] = []

        new_state["messages"].append({
            "role": "system",
            "content": f"Collected {len(news_data)} news articles about the Washington Nationals"
        })

    print("News agent completed. Next: blog_writer")
    return {"state": new_state, "next": "blog_writer"}


def blog_writer(state) -> AgentState:
    """
    Agent responsible for writing the initial blog post draft
    """
    print("Running blog_writer...")
    
    # Create a copy of the state to avoid modifying the input
    new_state = state.copy()

    print("\n==== DEBUG: Writing draft blog post ====")
    news_data = state.get("news_data", [])
    print(f"News data length: {len(news_data)}")
    
    # If no news data is available, get some with search_nationals_news
    if not news_data:
        news_data = search_nationals_news.invoke(None)
        new_state["news_data"] = news_data
        print(f"Retrieved {len(news_data)} news articles")
    
    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    # Print sample news data
    print(f"Sample news data: {json.dumps(news_data[0] if news_data else {})}")
    
    messages = [
        SystemMessage(
            content="You are a passionate baseball blogger who specializes in writing engaging content about the Washington Nationals."),
        HumanMessage(content=f"""
            Based on these recent news articles about the Washington Nationals:

            {json.dumps(news_data, indent=2)}

            Write an engaging blog post about their most recent game. Your blog post should:

            1. Have a catchy, attention-grabbing title
            2. Be approximately 500 words in length
            3. Focus on the key moments, plays, and standout players from the game
            4. Include the final score and important statistics
            5. Use an enthusiastic, fan-oriented tone that shows your passion for the team

            Write ONLY the complete blog post, formatted and ready for publication.
        """)
    ]

    # Generate the draft blog post
    response = llm.invoke(messages)
    draft_blog_post = response.content
    print(f"Draft blog post generated, length: {len(draft_blog_post)}")
    print(f"Preview: {draft_blog_post[:100]}...")
    
    # Update state
    new_state["draft_blog_post"] = draft_blog_post
    
    # Debug state update
    print(f"Updated state draft_blog_post length: {len(new_state['draft_blog_post'])}")

    # Add a log message
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "system",
        "content": "Draft blog post written"
    })
    
    print("Blog writer completed. Next: proofreader")
    return {"state": new_state, "next": "proofreader"}


def proofreader(state) -> AgentState:
    """
    Agent responsible for proofreading the blog post and providing feedback
    """
    print("Running proofreader...")
    
    # Make a copy of the state
    new_state = state.copy()

    print("\n==== DEBUG: Proofreading blog post ====")
    
    # Check if draft blog post is available
    if 'draft_blog_post' not in state:
        print("ERROR: Draft blog post not found in state. Returning to blog_writer.")
        return {"state": new_state, "next": "blog_writer"}
        
    print(f"Draft length: {len(state['draft_blog_post'])}")
    print(f"Draft preview: {state['draft_blog_post'][:100]}...")
    
    # Initialize the language model with lower temperature for more consistent proofreading
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.2
    )

    # Create messages for the LLM
    messages = [
        SystemMessage(content="You are a professional editor who specializes in proofreading sports content."),
        HumanMessage(content=f"""
            Proofread this blog post about a Washington Nationals baseball game:

            {state["draft_blog_post"]}

            Check the post against these news sources to ensure accuracy:

            {json.dumps(state.get("news_data", []), indent=2)}

            Evaluate the post for:
            1. Spelling and grammar errors
            2. Factual accuracy compared to the news sources
            3. Flow and readability
            4. Appropriate tone for a baseball blog
            5. Clarity and conciseness

            Provide detailed, actionable feedback for improvement.
        """)
    ]

    # Generate the proofreading feedback
    response = llm.invoke(messages)
    proofread_feedback = response.content

    # Update state
    new_state["proofread_feedback"] = proofread_feedback

    # Add a log message
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "system",
        "content": "Proofreading feedback provided"
    })

    print("Proofreader completed. Next: finalizer")
    return {"state": new_state, "next": "finalizer"}


def finalizer(state) -> AgentState:
    """
    Agent responsible for creating the final version of the blog post
    """
    print("Running finalizer...")
    
    # Make a copy of the state
    new_state = state.copy()

    print("\n==== DEBUG: Generating final post from feedback ====")
    
    # Check if required data is available
    if 'draft_blog_post' not in state:
        print("ERROR: Draft blog post not found in state. Returning to blog_writer.")
        return {"state": new_state, "next": "blog_writer"}
        
    if 'proofread_feedback' not in state:
        print("ERROR: Proofread feedback not found in state. Returning to proofreader.")
        return {"state": new_state, "next": "proofreader"}
    
    print(f"Draft blog post: {state['draft_blog_post'][:50]}...")
    print(f"Proofread feedback: {state['proofread_feedback'][:50]}...")
    
    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    # Create messages for the LLM
    messages = [
        SystemMessage(content="You are a skilled sports writer who incorporates feedback to improve blog posts."),
        HumanMessage(content=f"""
            You wrote this draft blog post about a Washington Nationals game:

            {state["draft_blog_post"]}

            You received this proofreading feedback:

            {state["proofread_feedback"]}

            Create a final, revised version of the blog post that addresses all the feedback while 
            maintaining your enthusiastic tone and fan-oriented style.

            Return ONLY the final, polished blog post ready for publication.
        """)
    ]

    # Generate the final blog post
    response = llm.invoke(messages)
    final_blog_post = response.content

    # Update state
    new_state["final_blog_post"] = final_blog_post

    # Add a log message
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "system",
        "content": "Final blog post written based on proofreading feedback"
    })

    print("Finalizer completed. Workflow ending.")
    return {"state": new_state, "next": END}


def build_nationals_blog_system() -> StateGraph:
    """
    Build and return the multi-agent workflow for creating Nationals blog posts using LangGraph
    """
    # Define the workflow as a graph
    workflow = StateGraph(AgentState)
    
    # Add the agent nodes
    workflow.add_node("news_agent", news_agent)
    workflow.add_node("blog_writer", blog_writer)
    workflow.add_node("proofreader", proofreader)
    workflow.add_node("finalizer", finalizer)
    
    # Define the edges between nodes
    workflow.add_edge("news_agent", "blog_writer")
    workflow.add_edge("blog_writer", "proofreader")
    workflow.add_edge("proofreader", "finalizer")
    workflow.add_edge("finalizer", END)
    
    # Set the entry point (start node)
    workflow.set_entry_point("news_agent")
    
    # Compile the graph
    return workflow.compile()


def run_nationals_blog_system_debug() -> str:
    """
    Run the multi-agent system in a simple sequential way for debugging
    """
    # Ensure Anthropic API key is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder value.")
        os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"

    # Get initial news data
    news_data = search_nationals_news.invoke(None)
    
    # Initialize the state with news data
    initial_state = {
        "messages": [{"role": "system", "content": "Starting blog creation workflow"}],
        "news_data": news_data
    }
    
    print("Starting debug workflow...")
    
    # Run news_agent
    print("\n=== NEWS AGENT ===")
    news_result = news_agent(initial_state)
    state = news_result["state"]
    
    # Run blog_writer
    print("\n=== BLOG WRITER ===")
    blog_result = blog_writer(state)
    state = blog_result["state"]
    
    # Run proofreader
    print("\n=== PROOFREADER ===")
    proof_result = proofreader(state)
    state = proof_result["state"]
    
    # Run finalizer
    print("\n=== FINALIZER ===")
    final_result = finalizer(state)
    state = final_result["state"]
    
    # Extract and return the final blog post
    final_blog_post = state.get("final_blog_post", "Failed to generate blog post.")
    
    print("\n=== FINAL BLOG POST ===\n")
    print(final_blog_post)
    
    return final_blog_post


def run_nationals_blog_system() -> str:
    """
    Run the multi-agent system and return the final blog post
    """
    # Ensure Anthropic API key is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder value.")
        os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"

    # Choose whether to use the LangGraph or sequential approach
    # For now, just run the debug (sequential) version that we know works
    return run_nationals_blog_system_debug()
    
    # The LangGraph version below can be uncommented after more debugging:
    """
    # Build the workflow
    workflow = build_nationals_blog_system()
    
    # Get initial news data
    news_data = search_nationals_news.invoke(None)
    
    # Initialize the state with news data
    initial_state = AgentState(
        messages=[{"role": "system", "content": "Starting blog creation workflow"}],
        news_data=news_data
    )
    
    # Execute the workflow
    print("Starting the Nationals blog creation workflow...")
    result = workflow.invoke(initial_state)
    
    # Extract and return the final blog post
    final_blog_post = result.get("final_blog_post", "Failed to generate blog post.")
    
    print("\n=== FINAL BLOG POST ===\n")
    print(final_blog_post)
    
    return final_blog_post
    """


if __name__ == "__main__":
    # Run the system
    run_nationals_blog_system()