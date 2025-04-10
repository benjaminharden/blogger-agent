import os
from typing import Dict, List, TypedDict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic  # Changed from langchain_openai to langchain_anthropic
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
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        return get_mock_nationals_news()


# Define the agents (nodes in the graph)
def news_agent(state: AgentState) -> Dict:
    """
    Agent responsible for searching news and writing a blog post
    """
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

        return {"state": new_state, "next": "blog_writer"}

    # If we have proofreading feedback, create the final version
    elif "proofread_feedback" in state and state["proofread_feedback"]:
        print("\n==== DEBUG: Generating final post from feedback ====")
        print(f"Draft blog post: {state.get('draft_blog_post', 'MISSING')[:50]}...")
        print(f"Proofread feedback: {state.get('proofread_feedback', 'MISSING')[:50]}...")
        
        # Initialize the language model (Claude instead of OpenAI)
        llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet model
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
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

        return {"state": new_state, "next": END}

    # If we're in an unexpected state, end the workflow
    return {"state": new_state, "next": "blog_writer"}


def blog_writer(state: AgentState) -> Dict:
    """
    Agent responsible for writing the initial blog post draft
    """
    # Make a copy of the state
    new_state = state.copy()

    print("\n==== DEBUG: Writing draft blog post ====")
    print(f"News data length: {len(state.get('news_data', []))}")
    
    # Initialize the language model (Claude instead of OpenAI)
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet model
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
        temperature=0.7
    )

    # Create messages for the LLM
    news_data = state.get("news_data", [])
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

    return {"state": new_state, "next": "proofreader"}


def proofreader(state: AgentState) -> Dict:
    """
    Agent responsible for proofreading the blog post and providing feedback
    """
    # Make a copy of the state
    new_state = state.copy()

    print("\n==== DEBUG: Proofreading blog post ====")
    print(f"Draft blog post available: {'draft_blog_post' in state}")
    if 'draft_blog_post' in state:
        print(f"Draft length: {len(state['draft_blog_post'])}")
        print(f"Draft preview: {state['draft_blog_post'][:100]}...")
    
    # Initialize the language model with lower temperature for more consistent proofreading
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet model
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
        temperature=0.2
    )

    # Create messages for the LLM
    messages = [
        SystemMessage(content="You are a professional editor who specializes in proofreading sports content."),
        HumanMessage(content=f"""
            Proofread this blog post about a Washington Nationals baseball game:

            {state["draft_blog_post"]}

            Check the post against these news sources to ensure accuracy:

            {json.dumps(state["news_data"], indent=2)}

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

    return {"state": new_state, "next": "news_agent"}


# Add a function to create a web search agent
def web_search_agent(state: AgentState) -> Dict:
    """
    Agent responsible for searching the web for the latest Washington Nationals news
    """
    # Make a copy of the state
    new_state = state.copy()

    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet model
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
        temperature=0.2
    )

    # In a real implementation, you would use LangChain's tools to perform a web search
    # For simplicity, we'll use our mock data
    news_data = search_nationals_news.invoke(None)

    # Update state with the search results
    new_state["news_data"] = news_data

    # Add a log message
    if "messages" not in new_state:
        new_state["messages"] = []

    new_state["messages"].append({
        "role": "system",
        "content": f"Found {len(news_data)} recent articles about Washington Nationals games"
    })

    return {"state": new_state, "next": "blog_writer"}


# Instead of attempting to fix the langgraph approach, we'll keep the simplified, working solution
# This is a placeholder that would be used in a graph-based approach
def build_nationals_blog_system() -> Any:
    """
    Build and return the multi-agent workflow for creating Nationals blog posts
    """
    # For now, we're using a simplified sequential approach that works correctly
    print("Using simplified sequential approach instead of graph workflow")
    return None


# Main function to run the system using a simplified sequential approach
def run_nationals_blog_system() -> str:
    """
    Run the multi-agent system and return the final blog post
    """
    # Ensure Anthropic API key is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder value.")
        os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

    # Get news data (using search or mock data)
    news_data = search_nationals_news.invoke(None)
    print(f"Got {len(news_data)} news articles")
    
    # Print out links to news articles
    print("\n=== NEWS ARTICLE LINKS ===")
    for i, article in enumerate(news_data):
        print(f"{i+1}. {article['title']}: {article['url']}")
    
    # Write draft blog post directly
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",  # Using Claude 3.7 Sonnet model
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
        temperature=0.7
    )
    
    # Create messages for the LLM
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
    print("Generating draft blog post...")
    response = llm.invoke(messages)
    draft_blog_post = response.content
    print(f"Draft blog post generated, length: {len(draft_blog_post)}")
    
    # Proofread the blog post
    print("Proofreading the blog post...")
    proofread_messages = [
        SystemMessage(content="You are a professional editor who specializes in proofreading sports content."),
        HumanMessage(content=f"""
            Proofread this blog post about a Washington Nationals baseball game:

            {draft_blog_post}

            Check the post against these news sources to ensure accuracy:

            {json.dumps(news_data, indent=2)}

            Evaluate the post for:
            1. Spelling and grammar errors
            2. Factual accuracy compared to the news sources
            3. Flow and readability
            4. Appropriate tone for a baseball blog
            5. Clarity and conciseness

            Provide detailed, actionable feedback for improvement.
        """)
    ]
    
    response = llm.invoke(proofread_messages)
    proofread_feedback = response.content
    print(f"Proofreading feedback generated, length: {len(proofread_feedback)}")
    
    # Generate final blog post based on feedback
    print("Generating final blog post...")
    final_messages = [
        SystemMessage(content="You are a skilled sports writer who incorporates feedback to improve blog posts."),
        HumanMessage(content=f"""
            You wrote this draft blog post about a Washington Nationals game:

            {draft_blog_post}

            You received this proofreading feedback:

            {proofread_feedback}

            Create a final, revised version of the blog post that addresses all the feedback while 
            maintaining your enthusiastic tone and fan-oriented style.

            Return ONLY the final, polished blog post ready for publication.
        """)
    ]
    
    response = llm.invoke(final_messages)
    final_blog_post = response.content
    print(f"Final blog post generated, length: {len(final_blog_post)}")
    
    print("\n=== FINAL BLOG POST ===\n")
    print(final_blog_post)
    
    return final_blog_post


if __name__ == "__main__":
    # Run the system
    run_nationals_blog_system()