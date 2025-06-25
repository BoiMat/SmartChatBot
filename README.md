# 🐕 Dog Behavior RAG System

An intelligent Retrieval-Augmented Generation (RAG) system specifically designed for dog behavior research. This system doesn't just search for papers - it intelligently analyzes your queries and asks targeted follow-up questions to help you find the most relevant academic research.

## Key Features

- **Web Interface**: Clean, intuitive Streamlit-based web UI for easy interaction
- **Interactive Query Refinement**: The system analyzes your question and asks targeted follow-up questions to improve search accuracy
- **Flexible Answer Modes**: Choose between direct answers or enhanced query refinement with clarifying questions
- **Semantic Search**: Uses advanced embedding models to find relevant papers based on meaning, not just keywords
- **Academic Paper Database**: Stores and searches through academic papers with structured metadata

## 🚀 What Makes This Special

Unlike traditional RAG systems that simply return results for any query, this system:

1. **Analyzes Query Context**: Detects specific behaviors, breeds, age groups, and study types mentioned
2. **Identifies Missing Information**: Determines what context might be missing from your query
3. **Asks Targeted Questions**: Generates specific follow-up questions to narrow down your search
4. **Refines Search**: Uses your answers to create a more targeted query for better results

## 📋 Example Interaction

```
User: "What causes aggression in dogs?"

System: "I detected you're asking about aggression. Let me ask a few questions:

1. What specific dog behavior are you most interested in?
   1. Aggression and dominance
   2. Anxiety and fear  
   3. Training and learning
   4. Social behavior
   5. Other

2. Are you interested in specific dog breeds, or general behavior?
   1. Specific breeds only
   2. All breeds (general behavior)
   3. Large breeds
   4. Small breeds

Refined query: "aggression in dogs specifically about aggression and dominance in large dog breeds"
```

## Installation

### Prerequisites
- Python 3.10+ installed
- Ollama running locally with llama3.2:3b model:
  ```bash
  # Install Ollama (if not installed)
  # Then pull the required model:
  ollama pull llama3.2:3b
  ```

### Setup
Install dependencies:
```bash
uv sync
```

## Usage

### Web Interface

Launch the Streamlit web interface:
```bash
uv run streamlit run web_interface.py
```

The web interface will open automatically in your browser at `http://localhost:8501`.

#### How to Use the Web Interface:

1. **Enter Your Question**: Type your dog behavior research question in the text area
2. **Choose Your Approach**:
   - **Default**: Click "Ask Question" for an immediate answer
   - **Enhanced**: Check "Improve Query" first, then click "Ask Question" to get clarifying questions for more targeted results
3. **Review Results**: Get comprehensive, research-based answers in an easy-to-read format
4. **Explore Sources**: View relevant research papers and citations in the sidebar
5. **Conversation History**: Track your previous questions and answers

## Architecture

### Core Components

1. **`database_schema.py`**: Defines the data models for academic papers
   - Paper metadata (title, authors, journal, etc.)
   - Dog-specific fields (breeds, behaviors, age ranges)
   - Behavior and breed categorization

2. **`query_refinement.py`**: The intelligence engine that analyzes queries
   - Extracts context (behaviors, breeds, ages, study types)
   - Generates targeted follow-up questions
   - Refines queries based on user responses

3. **`vector_database.py`**: Vector database interface using ChromaDB
   - Stores papers with semantic embeddings
   - Enables similarity search
   - Supports metadata filtering

5. **`main.py`**: Main application with rich console interface
   - Interactive conversation flow
   - Beautiful formatted output
   - Command handling

### Data Flow

1. User enters a query
2. Query Refinement Engine analyzes the query for context
3. If confidence is low, system asks follow-up questions
4. Refined query is used to search the vector database
5. Results are formatted and displayed with relevance scores

## Technical Details

- **Vector Embeddings**: Uses sentence-transformers 'nomic-ai/nomic-embed-text-v1' model
- **Database**: ChromaDB for persistent vector storage
- **Async Support**: Asyncio for responsive user interaction