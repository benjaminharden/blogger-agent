# Blogger Agent 🏟️⚾

A multi-agent system that automatically generates engaging blog posts about Washington Nationals baseball games using AI-powered workflow automation.

## 🌟 Features

- **Intelligent News Search**: Automatically searches for the latest Washington Nationals news using Google Custom Search API
- **Multi-Agent Workflow**: Employs specialized AI agents for different tasks:
  - 📰 **News Agent**: Collects recent game information
  - ✍️ **Blog Writer**: Drafts engaging content
  - 📝 **Proofreader**: Reviews for accuracy and quality
  - ✨ **Finalizer**: Polishes the final version
- **Powered by Claude 3.7 Sonnet**: Leverages Anthropic's latest AI model for high-quality content generation
- **Graceful Degradation**: Falls back to mock data if API credentials aren't configured
- **Type-Safe**: Built with Python type hints for better code quality

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐
│ news_agent  │ --> │ blog_writer │ --> │ proofreader │ --> │ finalizer │
└─────────────┘     └─────────────┘     └─────────────┘     └───────────┘
```

Each agent receives the workflow state, performs its specialized task, and passes the updated state to the next agent.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- (Optional) Google Custom Search API credentials for live news

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/blogger-agent.git
   cd blogger-agent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```bash
   # Required
   ANTHROPIC_API_KEY=sk-ant-your-key-here

   # Optional (uses mock data if not provided)
   GOOGLE_API_KEY=your-google-api-key
   GOOGLE_CSE_ID=your-custom-search-engine-id
   ```

### Running the Agent

```bash
python agent.py
```

The system will:
1. 🔍 Search for recent Washington Nationals news
2. 📄 Display found article links
3. ✍️ Generate a draft blog post (~500 words)
4. 🔍 Proofread the content
5. ✨ Create a final polished version
6. 📋 Print the final blog post to the console

## 📋 Example Output

```
=== SEARCH RESULTS ===
Found article 1: Nationals defeat Marlins 5-3 behind CJ Abrams' home run at ...
Found article 2: MacKenzie Gore strikes out 10 in win over Phillies at ...

Running news_agent...
Running blog_writer...
Draft blog post generated, length: 1847

Running proofreader...
Proofreader completed.

Running finalizer...

=== FINAL BLOG POST ===

Nationals Surge to Victory: CJ Abrams Powers Win Over Marlins

The Washington Nationals continued their impressive form with a commanding 5-3
victory over the Miami Marlins... [full blog post]
```

## 🛠️ Configuration

### Customizing the Workflow

You can modify agent behavior in `agent.py`:

- **Search query**: Line 59 - Adjust the search terms
- **Blog length**: Line 190 - Change word count requirement
- **Temperature settings**:
  - Lines 171, 318: 0.7 (creative writing)
  - Line 245: 0.2 (precise proofreading)

### Using Mock Data

If you don't have Google API credentials, the system automatically uses realistic mock data for demonstration. This is perfect for testing and development.

## 📚 Project Structure

```
blogger-agent/
├── agent.py              # Main application with all agents
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── CLAUDE.md            # AI assistant documentation
```

## 🔧 Technology Stack

- **LangChain**: Framework for building LLM applications
- **LangGraph**: Workflow orchestration
- **Anthropic Claude**: AI model for content generation
- **Google Custom Search**: Real-time news retrieval
- **Python 3.8+**: Core language

## 🧪 Development

### Type Checking

The project uses Python TypedDict for state management:

```python
class AgentState(TypedDict, total=False):
    messages: List[Dict[str, str]]
    news_data: List[NewsArticle]
    draft_blog_post: str
    proofread_feedback: str
    final_blog_post: str
```

### Adding New Agents

See `CLAUDE.md` for detailed instructions on extending the workflow with additional agents.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [x] ~~Enable the LangGraph workflow~~ ✅ **Completed (2025-12-14)**
- [ ] Add support for other MLB teams
- [ ] Implement blog post saving to files
- [ ] Add unit tests
- [ ] Add HTML/Markdown output formatting
- [ ] Create a web interface
- [ ] Add conditional routing in LangGraph for more complex workflows

## 📝 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Inspired by multi-agent system architectures

## 📞 Support

For issues or questions:
- Check `CLAUDE.md` for detailed technical documentation
- Review the code comments in `agent.py`
- Open an issue on GitHub

---

**Note**: This is a demonstration project showing how to build multi-agent workflows with LangChain and Claude. The project now uses LangGraph for proper workflow orchestration, demonstrating best practices for building production-ready multi-agent systems.
