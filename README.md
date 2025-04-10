# Blogger Agent

A multi-agent system that automatically generates engaging blog posts about Washington Nationals baseball games.

## Features

- Searches for the latest Washington Nationals news using Google Custom Search API
- Generates a draft blog post about recent games
- Proofreads the content for accuracy, grammar, and style
- Creates a final polished blog post incorporating feedback

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following credentials:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_google_custom_search_id
   ```

## Usage

Run the main script to generate a blog post:

```
python agent.py
```

The script will:
1. Search for recent Nationals news
2. Print the found article links
3. Generate a draft blog post
4. Proofread the content
5. Create a final polished version

## Requirements

- Python 3.8+
- LangChain
- Anthropic Claude API access (uses Claude 3.7 Sonnet model)
- Google Custom Search API credentials

## Note

If Google API credentials are not found in the `.env` file, the system will fall back to using mock data for demonstration purposes.