import os
import sys
import argparse
from typing import Dict, List, TypedDict, Any, Optional
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
    subject: str  # The topic/subject for the blog post
    messages: List[Dict[str, str]]
    news_data: List[NewsArticle]
    draft_blog_post: str
    proofread_feedback: str
    final_blog_post: str


# Tool for searching news on any subject
@tool
def search_news(subject: str) -> List[NewsArticle]:
    """
    Search for the latest news on a given subject.
    Returns recent articles about the subject.

    Args:
        subject: The topic to search for (e.g., "Washington Nationals baseball", "AI technology")
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
        return get_mock_news(subject)

    params = {
        "key": google_api_key,
        "cx": google_cse_id,
        "q": f"{subject} recent news from last 24 hours",
        "num": 5
    }

    try:
        response = requests.get(search_url, params=params, headers=headers)
        if response.status_code == 200:
            results = response.json().get("items", [])
            articles = []

            print(f"\n=== SEARCH RESULTS FOR: {subject} ===")
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
            return get_mock_news(subject)
    except Exception as e:
        print(f"Error during web search: {str(e)}")
        return get_mock_news(subject)


def get_mock_news(subject: str) -> List[NewsArticle]:
    """
    Return mock data for demonstration purposes based on the subject.

    Args:
        subject: The topic to generate mock data for
    """
    print(f"Using mock data for subject: {subject}")

    # Generate generic mock articles tailored to the subject
    return [
        {
            "title": f"Breaking: Major developments in {subject}",
            "source": "example-news.com",
            "url": f"https://example-news.com/{subject.replace(' ', '-').lower()}",
            "published_at": datetime.now().isoformat(),
            "summary": f"Recent analysis shows significant progress in {subject}, with experts weighing in on the latest developments."
        },
        {
            "title": f"Expert analysis: The future of {subject}",
            "source": "tech-digest.com",
            "url": f"https://tech-digest.com/{subject.replace(' ', '-').lower()}",
            "published_at": datetime.now().isoformat(),
            "summary": f"Industry leaders discuss the evolving landscape of {subject} and what it means for the future."
        },
        {
            "title": f"Top insights on {subject} this week",
            "source": "news-today.com",
            "url": f"https://news-today.com/{subject.replace(' ', '-').lower()}",
            "published_at": datetime.now().isoformat(),
            "summary": f"A comprehensive look at the most important developments and trends in {subject}."
        }
    ]


# Define the agents (nodes in the graph)
def news_agent(state: AgentState) -> AgentState:
    """
    Agent responsible for searching news on the specified subject
    """
    print("Running news_agent...")

    subject = state.get("subject", "general news")
    print(f"Searching for news about: {subject}")

    # Search for news if we don't have it yet
    if "news_data" not in state or not state["news_data"]:
        # Get the news using the subject
        news_data = search_news.invoke(subject)

        # Prepare messages list
        messages = state.get("messages", [])
        messages.append({
            "role": "system",
            "content": f"Collected {len(news_data)} news articles about {subject}"
        })

        print(f"News agent completed. Collected {len(news_data)} articles.")
        # Return state updates for LangGraph
        return {
            "news_data": news_data,
            "messages": messages
        }

    print("News agent completed (data already present).")
    return {}


def blog_writer(state: AgentState) -> AgentState:
    """
    Agent responsible for writing the initial blog post draft
    """
    print("Running blog_writer...")

    subject = state.get("subject", "general topic")
    print(f"\n==== DEBUG: Writing draft blog post about {subject} ====")
    news_data = state.get("news_data", [])
    print(f"News data length: {len(news_data)}")

    # If no news data is available, get some with search_news
    if not news_data:
        news_data = search_news.invoke(subject)
        print(f"Retrieved {len(news_data)} news articles")

    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    # Print sample news data
    print(f"Sample news data: {json.dumps(news_data[0] if news_data else {})}")

    llm_messages = [
        SystemMessage(
            content="You are an experienced and engaging blogger who specializes in writing compelling content on various topics."),
        HumanMessage(content=f"""
            Based on these recent news articles about {subject}:

            {json.dumps(news_data, indent=2)}

            Write an engaging blog post about {subject}. Your blog post should:

            1. Have a catchy, attention-grabbing title
            2. Be approximately 500 words in length
            3. Focus on the key insights, developments, and important points from the news
            4. Include relevant facts, statistics, or quotes if mentioned in the sources
            5. Use an enthusiastic, engaging tone that draws readers in
            6. Provide context and explain why this topic matters

            Write ONLY the complete blog post, formatted and ready for publication.
        """)
    ]

    # Generate the draft blog post
    response = llm.invoke(llm_messages)
    draft_blog_post = response.content
    print(f"Draft blog post generated, length: {len(draft_blog_post)}")
    print(f"Preview: {draft_blog_post[:100]}...")

    # Prepare messages list
    messages = state.get("messages", [])
    messages.append({
        "role": "system",
        "content": f"Draft blog post written about {subject}"
    })

    print("Blog writer completed.")
    # Return state updates for LangGraph
    return {
        "draft_blog_post": draft_blog_post,
        "messages": messages
    }


def proofreader(state: AgentState) -> AgentState:
    """
    Agent responsible for proofreading the blog post and providing feedback
    """
    print("Running proofreader...")

    subject = state.get("subject", "the topic")
    print(f"\n==== DEBUG: Proofreading blog post about {subject} ====")

    # Check if draft blog post is available
    if 'draft_blog_post' not in state or not state['draft_blog_post']:
        print("ERROR: Draft blog post not found in state.")
        raise ValueError("Draft blog post is required for proofreading")

    print(f"Draft length: {len(state['draft_blog_post'])}")
    print(f"Draft preview: {state['draft_blog_post'][:100]}...")

    # Initialize the language model with lower temperature for more consistent proofreading
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.2
    )

    # Create messages for the LLM
    llm_messages = [
        SystemMessage(content="You are a professional editor who specializes in proofreading and improving blog content."),
        HumanMessage(content=f"""
            Proofread this blog post about {subject}:

            {state["draft_blog_post"]}

            Check the post against these news sources to ensure accuracy:

            {json.dumps(state.get("news_data", []), indent=2)}

            Evaluate the post for:
            1. Spelling and grammar errors
            2. Factual accuracy compared to the news sources
            3. Flow and readability
            4. Appropriate tone for an engaging blog post
            5. Clarity and conciseness
            6. Proper use of context and explanations

            Provide detailed, actionable feedback for improvement.
        """)
    ]

    # Generate the proofreading feedback
    response = llm.invoke(llm_messages)
    proofread_feedback = response.content

    # Prepare messages list
    messages = state.get("messages", [])
    messages.append({
        "role": "system",
        "content": f"Proofreading feedback provided for {subject} post"
    })

    print("Proofreader completed.")
    # Return state updates for LangGraph
    return {
        "proofread_feedback": proofread_feedback,
        "messages": messages
    }


def finalizer(state: AgentState) -> AgentState:
    """
    Agent responsible for creating the final version of the blog post
    """
    print("Running finalizer...")

    subject = state.get("subject", "the topic")
    print(f"\n==== DEBUG: Generating final post about {subject} ====")

    # Check if required data is available
    if 'draft_blog_post' not in state or not state['draft_blog_post']:
        print("ERROR: Draft blog post not found in state.")
        raise ValueError("Draft blog post is required for finalizer")

    if 'proofread_feedback' not in state or not state['proofread_feedback']:
        print("ERROR: Proofread feedback not found in state.")
        raise ValueError("Proofread feedback is required for finalizer")

    print(f"Draft blog post: {state['draft_blog_post'][:50]}...")
    print(f"Proofread feedback: {state['proofread_feedback'][:50]}...")

    # Initialize the language model
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

    # Create messages for the LLM
    llm_messages = [
        SystemMessage(content="You are a skilled writer who incorporates editorial feedback to improve blog posts."),
        HumanMessage(content=f"""
            You wrote this draft blog post about {subject}:

            {state["draft_blog_post"]}

            You received this proofreading feedback:

            {state["proofread_feedback"]}

            Create a final, revised version of the blog post that addresses all the feedback while
            maintaining an enthusiastic and engaging tone.

            Return ONLY the final, polished blog post ready for publication.
        """)
    ]

    # Generate the final blog post
    response = llm.invoke(llm_messages)
    final_blog_post = response.content

    # Prepare messages list
    messages = state.get("messages", [])
    messages.append({
        "role": "system",
        "content": f"Final blog post written about {subject}"
    })

    print("Finalizer completed. Workflow ending.")
    # Return state updates for LangGraph
    return {
        "final_blog_post": final_blog_post,
        "messages": messages
    }


def build_blog_system() -> StateGraph:
    """
    Build and return the multi-agent workflow for creating blog posts using LangGraph.
    Works for any subject/topic.
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


# Legacy sequential debug function - no longer needed with LangGraph enabled
# Kept for reference only. LangGraph now handles the workflow orchestration.
# def run_nationals_blog_system_debug() -> str:
#     """
#     Run the multi-agent system in a simple sequential way for debugging
#     (DEPRECATED: LangGraph is now enabled and working)
#     """
#     pass


def run_blog_system(subject: str) -> str:
    """
    Run the multi-agent blog system using LangGraph and return the final blog post.

    Args:
        subject: The topic/subject for the blog post (e.g., "artificial intelligence",
                 "Washington Nationals baseball", "climate change")

    Returns:
        The final polished blog post as a string
    """
    # Ensure Anthropic API key is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY environment variable not set. Using placeholder value.")
        os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"

    # Build the LangGraph workflow
    workflow = build_blog_system()

    # Initialize the state with subject and starting message
    initial_state: AgentState = {
        "subject": subject,
        "messages": [{"role": "system", "content": f"Starting blog creation workflow for: {subject}"}]
    }

    # Execute the workflow
    print(f"Starting the blog creation workflow with LangGraph...")
    print(f"Subject: {subject}")
    print("=" * 60)
    result = workflow.invoke(initial_state)

    # Extract and return the final blog post
    final_blog_post = result.get("final_blog_post", "Failed to generate blog post.")

    print("\n" + "=" * 60)
    print("=== FINAL BLOG POST ===\n")
    print(final_blog_post)
    print("=" * 60)

    return final_blog_post


def get_subject_from_user() -> str:
    """
    Get the blog subject from the user via command-line arguments or interactive input.

    Returns:
        The subject for the blog post
    """
    parser = argparse.ArgumentParser(
        description="Generate an AI-powered blog post on any subject using multi-agent workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --subject "artificial intelligence trends"
  python agent.py -s "Washington Nationals baseball"
  python agent.py  (interactive mode)

The system will:
  1. Search for recent news about your subject
  2. Generate a draft blog post
  3. Proofread and provide feedback
  4. Create a final polished version
        """
    )

    parser.add_argument(
        '-s', '--subject',
        type=str,
        help='The subject/topic for the blog post (e.g., "climate change", "AI technology")'
    )

    args = parser.parse_args()

    # If subject provided via command line, use it
    if args.subject:
        return args.subject

    # Otherwise, prompt interactively
    print("\n" + "=" * 60)
    print("Welcome to the AI Blog Generator!")
    print("=" * 60)
    print("\nThis system will create a blog post about any subject you choose.")
    print("It uses a multi-agent workflow to:")
    print("  1. Search for recent news")
    print("  2. Draft a blog post")
    print("  3. Proofread the content")
    print("  4. Create a polished final version\n")

    subject = input("Enter the subject for your blog post: ").strip()

    if not subject:
        print("No subject provided. Using default: 'Washington Nationals baseball'")
        return "Washington Nationals baseball"

    return subject


if __name__ == "__main__":
    # Get the subject from user
    subject = get_subject_from_user()

    # Run the blog system with the provided subject
    run_blog_system(subject)