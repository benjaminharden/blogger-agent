# CLAUDE.md - AI Assistant Guide for Blogger Agent

## Project Overview

**Blogger Agent** is a multi-agent system built with Python, LangChain, and LangGraph that automatically generates engaging blog posts on **any subject**. The system uses Claude 3.7 Sonnet (via Anthropic API) to power a LangGraph workflow of specialized agents.

### Purpose
Generate publication-ready blog posts by:
1. Accepting a user-specified subject/topic
2. Searching for recent news about that subject
3. Drafting an engaging blog post
4. Proofreading for accuracy and quality
5. Producing a polished final version

### Technology Stack
- **Language**: Python 3.8+
- **AI Framework**: LangChain + LangGraph
- **LLM**: Claude 3.7 Sonnet (model: `claude-3-7-sonnet-20250219`)
- **External APIs**:
  - Anthropic API (Claude)
  - Google Custom Search API
- **Environment Management**: python-dotenv

---

## Repository Structure

```
blogger-agent/
├── agent.py              # Main application - contains all agents and workflow logic
├── README.md             # User-facing documentation
├── CLAUDE.md             # This file - AI assistant guidance
└── .env                  # Environment variables (not in repo, create locally)
```

### Key Files

#### agent.py (476 lines)
The entire application is contained in this single, well-structured file:

**Type Definitions** (lines 16-31):
- `NewsArticle`: TypedDict for news article structure
- `AgentState`: TypedDict for workflow state management

**Tools** (lines 36-131):
- `search_news(subject)`: Google Custom Search API integration with subject parameter
- `get_mock_news(subject)`: Subject-aware mock data provider for demos/testing

**Agent Functions** (lines 134-365):
- `news_agent(state)`: Searches and collects news articles about the subject
- `blog_writer(state)`: Generates initial blog draft tailored to the subject
- `proofreader(state)`: Reviews draft for accuracy, grammar, and style
- `finalizer(state)`: Creates polished final version incorporating feedback

**Workflow and Entry Points** (lines 368-508):
- `build_blog_system()`: LangGraph workflow definition (active)
- `run_blog_system(subject)`: Main entry point that executes the workflow
- `get_subject_from_user()`: CLI argument parser and interactive prompt handler

---

## Workflow Architecture

### Multi-Agent Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐
│ news_agent  │ --> │ blog_writer │ --> │ proofreader │ --> │ finalizer │
└─────────────┘     └─────────────┘     └─────────────┘     └───────────┘
      │                    │                    │                   │
  Searches news      Drafts post        Reviews draft      Final polish
  Google API         Claude 0.7 temp    Claude 0.2 temp    Claude 0.7 temp
```

### Agent Details

1. **news_agent** (lines 122-147)
   - Invokes `search_nationals_news()` tool
   - Stores results in `state["news_data"]`
   - Fallback to mock data if API fails
   - Returns: state with news articles

2. **blog_writer** (lines 150-221)
   - Input: `state["news_data"]`
   - Claude temperature: 0.7 (creative)
   - Generates ~500 word blog post
   - Output: `state["draft_blog_post"]`

3. **proofreader** (lines 224-290)
   - Input: `state["draft_blog_post"]` + `state["news_data"]`
   - Claude temperature: 0.2 (consistent/precise)
   - Checks: grammar, facts, flow, tone, clarity
   - Output: `state["proofread_feedback"]`

4. **finalizer** (lines 293-359)
   - Input: `state["draft_blog_post"]` + `state["proofread_feedback"]`
   - Claude temperature: 0.7 (creative)
   - Incorporates all feedback
   - Output: `state["final_blog_post"]`

### State Management

The `AgentState` TypedDict tracks:
- `messages`: List of system log messages
- `news_data`: List of NewsArticle objects
- `draft_blog_post`: Initial blog content
- `proofread_feedback`: Editorial feedback
- `final_blog_post`: Polished final version

Each agent:
1. Receives current state
2. Creates a copy (`state.copy()`)
3. Performs its work
4. Returns updated state + next node

---

## Environment Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Required for all functionality
ANTHROPIC_API_KEY=sk-ant-...

# Optional - falls back to mock data if missing
GOOGLE_API_KEY=AIza...
GOOGLE_CSE_ID=...
```

### Graceful Degradation

- Missing Google credentials → Uses `get_mock_nationals_news()` (lines 93-118)
- Missing Anthropic key → Warning printed, placeholder used (line 395)
- Search API failure → Falls back to mock data (lines 84-90)

---

## Code Patterns & Conventions

### 1. Error Handling
```python
# Pattern: Try external API, fall back to mock data
try:
    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code == 200:
        # Process results
    else:
        return get_mock_nationals_news()
except Exception as e:
    print(f"Error during web search: {str(e)}")
    return get_mock_nationals_news()
```

### 2. Agent Pattern
```python
def agent_name(state: AgentState) -> AgentState:
    """Agent description"""
    print("Running agent_name...")

    # Do work...
    result_value = perform_work(state)

    # Prepare messages list
    messages = state.get("messages", [])
    messages.append({
        "role": "system",
        "content": "Agent completed its work"
    })

    print("Agent completed.")
    # Return state updates for LangGraph (not the full state)
    return {
        "key": result_value,
        "messages": messages
    }
```

### 3. LLM Invocation
```python
llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    temperature=0.7  # 0.2 for proofreading, 0.7 for creative
)

messages = [
    SystemMessage(content="Role description"),
    HumanMessage(content=f"Task with {context}")
]

response = llm.invoke(messages)
result = response.content
```

### 4. Debug Printing
Extensive debug output throughout:
- `print("\n==== DEBUG: Section ====")` for major steps
- `print(f"Variable: {value}")` for state tracking
- Helps with troubleshooting workflow

---

## Development Workflows

### Running the System

```bash
# Interactive mode (prompts for subject)
python agent.py

# Command-line argument
python agent.py --subject "artificial intelligence"
python agent.py -s "climate change"

# Get help
python agent.py --help

# Expected output flow:
# 1. Subject prompt or parsing
# 2. "Running news_agent..."
# 3. "Searching for news about: [subject]"
# 4. Google search results or "Using mock data"
# 5. "Running blog_writer..."
# 6. "Draft blog post generated"
# 7. "Running proofreader..."
# 8. "Proofreader completed"
# 9. "Running finalizer..."
# 10. "=== FINAL BLOG POST ===" + content
```

### Current Implementation

The system now uses **LangGraph for workflow orchestration** (enabled as of 2025-12-14):

**LangGraph Implementation (lines 347-370)**: ✅ Active
- Proper graph-based execution with StateGraph
- Agents communicate via state updates
- Automatic workflow orchestration
- Used via `run_nationals_blog_system()` (lines 383-413)

**Key Changes:**
- All agents now return state updates as dictionaries (not wrapped with "next" keys)
- LangGraph handles routing between nodes automatically
- The sequential debug version has been deprecated and removed
- Agents use proper type hints: `def agent_name(state: AgentState) -> AgentState`

---

## Making Changes

### Adding a New Agent

1. **Define the agent function** (follow pattern at lines 122-148):
```python
def new_agent(state: AgentState) -> AgentState:
    """Description of what this agent does"""
    print("Running new_agent...")

    # Your logic here
    result = perform_some_work(state)

    # Prepare messages list (always get from state first)
    messages = state.get("messages", [])
    messages.append({
        "role": "system",
        "content": "What this agent did"
    })

    print("New agent completed.")
    # Return ONLY the state updates (LangGraph merges with existing state)
    return {
        "new_key": result,
        "messages": messages
    }
```

2. **Update AgentState TypedDict** (lines 24-30):
```python
class AgentState(TypedDict, total=False):
    # ... existing fields
    new_key: str  # Add your new state field
```

3. **Add to LangGraph workflow** (lines 347-370):
```python
def build_nationals_blog_system() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add your new node
    workflow.add_node("new_agent", new_agent)

    # Add edges to connect it to the workflow
    workflow.add_edge("previous_agent", "new_agent")
    workflow.add_edge("new_agent", "next_agent")

    return workflow.compile()
```

### Modifying Search Behavior

Edit `search_nationals_news()` function (lines 34-90):
- **Query**: Change line 59 search query
- **Result count**: Change `"num": 5` parameter (line 60)
- **Source filtering**: Modify params dict (lines 56-61)

### Changing Blog Requirements

Edit blog_writer's prompt (lines 182-196):
- Word count requirement (line 190)
- Content focus (lines 191-193)
- Tone instructions (line 194)

### Adjusting Proofreading Criteria

Edit proofreader's prompt (lines 262-269):
- Add/remove evaluation criteria
- Change emphasis on specific aspects

---

## Testing & Validation

### Manual Testing

1. **Test with mock data** (no API keys needed):
```bash
# Don't set GOOGLE_API_KEY in .env
python agent.py
```
Expected: System uses mock Washington Nationals news

2. **Test with real search**:
```bash
# Set all API keys in .env
python agent.py
```
Expected: Real Google search results printed with URLs

### Validation Points

Check these in output:
- ✓ "Found article 1: ... at ..." (search results)
- ✓ "Draft blog post generated, length: ..." (draft created)
- ✓ "Draft length: ..." (proofreader received draft)
- ✓ "=== FINAL BLOG POST ===" (final output)

### Common Issues

**"Draft blog post not found in state"**
- Cause: State not properly passed between agents
- Location: proofreader() line 236-238
- Fix: Check state copying in blog_writer()

**"Warning: Google API credentials not found"**
- Not an error - system falls back to mock data
- Set `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` to resolve

**Empty/missing final_blog_post**
- Check: Each agent's return statement
- Verify: State updates aren't being overwritten
- Debug: Enable prints in finalizer (line 302 onwards)

---

## Dependencies

### Required Python Packages

While no `requirements.txt` exists, the code imports:

```python
# Standard library
import os
import json
from typing import Dict, List, TypedDict, Any
from datetime import datetime

# External packages (need pip install)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import requests
```

### Suggested requirements.txt

```txt
langchain>=0.1.0
langchain-core>=0.1.0
langchain-anthropic>=0.1.0
langgraph>=0.0.20
python-dotenv>=1.0.0
requests>=2.31.0
anthropic>=0.18.0
```

**Note**: Create this file if setting up a new environment.

---

## AI Assistant Guidelines

### When Debugging Issues

1. **Check state flow**: Verify each agent receives expected state keys
2. **Examine debug output**: Look for "DEBUG:" sections in console
3. **Validate API keys**: Ensure `.env` file is loaded (`load_dotenv()` at line 13)
4. **Test incrementally**: Run agents one at a time in `run_nationals_blog_system_debug()`

### When Adding Features

1. **Preserve agent pattern**: Always copy state, never mutate
2. **Add debug logging**: Print statements are encouraged
3. **Update type definitions**: Keep AgentState TypedDict in sync
4. **Test both workflows**: If re-enabling LangGraph, test both paths

### When Refactoring

**Do**:
- Extract shared LLM configuration (temperature varies by agent)
- Create helper functions for common patterns
- Add type hints to all functions (now using `state: AgentState`)
- Document agent responsibilities
- Return only state updates from agents (LangGraph merges them)

**Don't**:
- Remove debug prints (they're intentional)
- Return full copied state (return updates only)
- Change agent signatures without updating workflow
- Modify mock data structure without updating NewsArticle TypedDict
- Use old pattern of returning `{"state": ..., "next": ...}`

### Code Style Observations

- **Line length**: Generally reasonable, some long strings in prompts
- **Documentation**: Good docstrings on agents and tools
- **Error messages**: Informative with context
- **Type safety**: Uses TypedDict for state management

---

## Git Workflow

### Branch Strategy
- Main development branch: `claude/claude-md-mi85vpwcxu2ctoyo-01Xpxn3TbyEozbBWVQWaRQJS`
- Always create feature branches from current development branch
- Branch naming: Use descriptive names like `feature/new-agent` or `fix/state-bug`

### Commit Guidelines

```bash
# Stage changes
git add file.py

# Commit with descriptive message
git commit -m "Add sentiment analysis agent to workflow"

# Push to development branch
git push -u origin claude/claude-md-mi85vpwcxu2ctoyo-01Xpxn3TbyEozbBWVQWaRQJS
```

### Before Committing

1. **Test the workflow**: Run `python agent.py` successfully
2. **Check for secrets**: Never commit `.env` files
3. **Verify imports**: Ensure all dependencies are documented
4. **Update documentation**: Reflect changes in README.md or this file

---

## Future Improvements

### Suggested Enhancements

1. ~~**Enable LangGraph workflow**~~ ✅ **COMPLETED (2025-12-14)**
   - ~~Debug state passing issues~~
   - ~~Replace sequential execution~~
   - All agents now use LangGraph orchestration

2. ~~**Add requirements.txt**~~ ✅ **COMPLETED (2025-12-14)**
   - ~~Document exact dependency versions~~
   - ~~Enable easy environment setup~~

3. **Separate concerns**
   - Move agents to `agents/` directory
   - Extract tools to `tools/` directory
   - Configuration in `config.py`

4. **Add unit tests**
   - Test each agent independently
   - Mock LLM responses
   - Validate state transformations

5. **Enhanced error handling**
   - Retry logic for API calls
   - Better error messages
   - Logging framework instead of prints

6. **Output formatting**
   - Save blog posts to files
   - Add metadata (timestamp, sources)
   - Support multiple output formats (Markdown, HTML)

7. **Configuration options**
   - Make team name configurable (not just Nationals)
   - Adjustable word count
   - Tone customization

---

## Quick Reference

### File Locations
- Main logic: `agent.py`
- Environment vars: `.env` (create locally)
- Documentation: `README.md`, `CLAUDE.md`

### Key Functions
- Entry point: `run_blog_system(subject)` (line 405)
- User input handler: `get_subject_from_user()` (line 447)
- LangGraph workflow builder: `build_blog_system()` (line 368)
- Search: `search_news(subject)` (line 38)
- Mock data: `get_mock_news(subject)` (line 99)
- Agents: `news_agent()` (135), `blog_writer()` (167), `proofreader()` (235), `finalizer()` (302)

### Configuration
- Claude model: Lines 184, 253, 324 (`claude-3-7-sonnet-20250219`)
- Search query: Line 65 (uses subject parameter)
- Blog length: Line 204 (~500 words)
- Temperatures: 0.7 (creative at lines 187, 327), 0.2 (proofreading at line 256)
- CLI arguments: argparse setup at line 454

### State Keys
- `subject`: str (the topic for the blog post)
- `news_data`: List[NewsArticle]
- `draft_blog_post`: str
- `proofread_feedback`: str
- `final_blog_post`: str
- `messages`: List[Dict[str, str]]

---

## Support & Resources

### LangChain Documentation
- LangChain: https://python.langchain.com/docs/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Anthropic: https://docs.anthropic.com/claude/reference/

### Google Custom Search
- API Console: https://console.developers.google.com/
- Custom Search Engine: https://programmablesearchengine.google.com/

### Project Context
This is a demonstration/educational project showing multi-agent workflows with LangChain and Claude. The sequential debugging approach is intentionally kept simple for clarity.

---

## Recent Updates

### 2025-12-14 (Part 3) - Universal Subject Support! 🌐
- ✅ **Made system subject-agnostic** - Now works for ANY topic, not just baseball
- ✅ Added `subject` field to AgentState
- ✅ Renamed `search_nationals_news()` to `search_news(subject)`
- ✅ Updated `get_mock_news(subject)` to generate subject-aware mock data
- ✅ All agents now use dynamic subject from state
- ✅ Added CLI argument parsing (`--subject` / `-s`)
- ✅ Added interactive mode with user prompts
- ✅ Renamed `build_nationals_blog_system()` to `build_blog_system()`
- ✅ Created `get_subject_from_user()` for flexible input handling
- ✅ Updated all agent prompts to be topic-agnostic
- ✅ Expanded from 418 to 507 lines (+89 lines for new functionality)

### 2025-12-14 (Part 2) - LangGraph Enabled! 🎉
- ✅ **Enabled LangGraph workflow** - Now using proper graph-based execution
- ✅ Fixed all agent return signatures to work with LangGraph
- ✅ Agents now return state updates only (not full state copies)
- ✅ Added proper type hints: `def agent_name(state: AgentState) -> AgentState`
- ✅ Removed deprecated sequential debug workflow
- ✅ Updated all documentation to reflect LangGraph implementation
- ✅ Reduced agent.py from 476 to 418 lines (cleaner code)

### 2025-12-14 (Part 1) - Project Infrastructure
- ✅ Added `requirements.txt` with all Python dependencies
- ✅ Created `.env.example` template for environment variables
- ✅ Added comprehensive `.gitignore` for Python projects
- ✅ Completely rewrote README.md with better structure and examples
- ✅ Added emojis and visual improvements to documentation
- ✅ Included architecture diagram and example output

---

**Last Updated**: 2025-12-14
**Claude Model Used**: Claude 3.7 Sonnet (claude-3-7-sonnet-20250219)
**Project Status**: ✅ Fully functional with LangGraph workflow orchestration enabled
