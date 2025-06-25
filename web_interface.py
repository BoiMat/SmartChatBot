"""
Streamlit Web Interface for Dog Behavior RAG System
"""
import streamlit as st
import asyncio
import json
import logging
import threading
from typing import Dict, List, Any
from pathlib import Path
import sys
import os

# Add project root to path if needed
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Dog Behavior Research Assistant",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.answer-box {
    background-color: #e8f4fd;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    color: #000000 !important;
}
.answer-box * {
    color: #000000 !important;
}
.paper-box {
    background-color: #f8f9fa;
    padding: 0.8rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
    border-left: 3px solid #28a745;
    color: #000000 !important;
}
.paper-box strong {
    color: #000000 !important;
}
.paper-box * {
    color: #000000 !important;
}
.stAlert > div {
    background-color: #fff3cd;
    border-color: #ffeaa7;
    color: #856404;
}
.element-container {
    margin-bottom: 0.5rem !important;
}
/* Hide dividers and reduce spacing */
hr {
    display: none;
}
/* Style containers to look like question boxes */
.stContainer > div {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def run_async(coro):
    """Helper function to run async coroutines in Streamlit."""
    try:
        # Try to get the existing loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop for this thread
            result = None
            exception = None
            
            def run_in_thread():
                nonlocal result, exception
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    result = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e:
                    exception = e
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            
            if exception:
                raise exception
            return result
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_questions' not in st.session_state:
    st.session_state.current_questions = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'question_answers' not in st.session_state:
    st.session_state.question_answers = {}

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching."""
    try:
        # Import here to avoid early loading issues
        from llm_service import LLMDynamicRAG
        from vector_database import VectorDatabase
        
        vector_db = VectorDatabase(
            db_path="db_nomic",
            collection_name="dog_behavior",
            embedding_model="nomic-ai/nomic-embed-text-v1",
            chunk_size=2
        )
        
        rag = LLMDynamicRAG(
            vector_db=vector_db,
            query_analyzer_model="llama3.2:3b",
            answer_model="llama3.2:3b",
        )
        
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        logger.error(f"RAG initialization error: {e}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🐕 Dog Behavior Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        if st.button("🔄 Initialize/Refresh RAG System"):
            st.session_state.rag_system = initialize_rag_system()
        
        if st.session_state.rag_system is None:
            st.session_state.rag_system = initialize_rag_system()
        
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
            
            # Database stats
            try:
                paper_count = len(st.session_state.rag_system.vector_db.paper_dicts)
                st.info(f"📚 {paper_count} papers loaded")
                
                chunk_count = st.session_state.rag_system.vector_db.collection.count()
                st.info(f"📄 {chunk_count} text chunks")
            except Exception as e:
                st.warning(f"Could not fetch database stats: {str(e)}")
        else:
            st.error("❌ RAG System Not Available")
            st.info("Please check that Ollama is running and your database is set up correctly.")
            st.stop()
        
        # Search parameters
        st.subheader("🔍 Search Settings")
        top_k = st.slider("Number of documents to retrieve", 5, 100, 30)
          # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.current_questions = []
            st.session_state.analysis_result = None
            st.session_state.question_answers = {}
            st.rerun()
      # Main content area
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.header("💬 Ask Your Question")
          # Query input
        user_query = st.text_area(
            "Enter your dog behavior research question:",
            placeholder="e.g., Why does my dog bark so much when I leave the house?",
            height=100
        )
        
        # --- IMPROVED: Single button, toggle for improved query ---
        improve_query = st.checkbox(
            "🔍 Improve Query (ask clarifying questions)",
            value=False,
            help="If checked, you will be asked clarifying questions before getting an answer."
        )
        
        if user_query.strip():
            ask_button = st.button(
                "🚀 Ask Question",
                type="primary",
                use_container_width=True
            )
            if ask_button:
                if improve_query:
                    with st.spinner("Analyzing your question to generate clarifying questions..."):
                        try:
                            analysis_result = run_async(
                                st.session_state.rag_system.process_query(user_query, top_k)
                            )
                            st.session_state.analysis_result = analysis_result
                            st.session_state.current_questions = analysis_result.get('questions', [])
                            st.session_state.question_answers = {}  # Clear previous answers
                            st.session_state.conversation_history.append({
                                'type': 'query',
                                'content': user_query,
                                'timestamp': str(len(st.session_state.conversation_history))
                            })
                            if st.session_state.current_questions:
                                st.success(f"Generated {len(st.session_state.current_questions)} clarifying questions to improve your results!")
                            else:
                                st.info("No clarifying questions needed for this query. You can get a direct answer instead.")
                        except Exception as e:
                            st.error(f"Error analyzing query: {str(e)}")
                            logger.error(f"Query analysis error: {e}")
                else:
                    with st.spinner("Generating answer..."):
                        try:
                            answer_result = run_async(
                                st.session_state.rag_system.generate_direct_answer(
                                    user_query,
                                    top_k
                                )
                            )
                            st.markdown("### 📋 Research-Based Answer")
                            st.markdown(
                                f'<div class="answer-box">{answer_result["answer"]}</div>',
                                unsafe_allow_html=True
                            )
                            st.session_state.conversation_history.append({
                                'type': 'query',
                                'content': user_query,
                                'timestamp': str(len(st.session_state.conversation_history))
                            })
                            st.session_state.conversation_history.append({
                                'type': 'answer',
                                'content': answer_result["answer"],
                                'improved_query': user_query,  # Same as original for direct answer
                                'sources': answer_result.get('results', [])[:5]
                            })
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                            logger.error(f"Direct answer error: {e}")
        else:
            st.info("💡 **Tip:** Enter your question above, then click 'Ask Question' for a direct answer, or enable 'Improve Query' to refine your question first.")
        # --- END IMPROVED ---
        
        # Display clarifying questions if available
        if st.session_state.current_questions:
            st.subheader("🤔 Clarifying Questions")
            st.info("To provide more accurate results, please answer these questions:")
            user_answers = {}
            
            for i, question in enumerate(st.session_state.current_questions):
                with st.container():
                    st.write(f"**{i+1}. {question.question}**")
                    
                    if question.options:
                        # Multiple choice question with toggle buttons (allow multiple selections)
                        # Create columns for better layout of buttons
                        if len(question.options) <= 3:
                            cols = st.columns(len(question.options))
                        else:
                            # For more than 3 options, use 2 columns
                            cols = st.columns(2)
                        
                        # Initialize question answers if not exists
                        if question.question not in st.session_state.question_answers:
                            st.session_state.question_answers[question.question] = []
                        
                        current_selections = st.session_state.question_answers[question.question]
                        
                        for idx, option in enumerate(question.options):
                            col_idx = idx % len(cols)
                            with cols[col_idx]:
                                # Check if this option is currently selected
                                is_selected = option in current_selections
                                button_type = "primary" if is_selected else "secondary"
                                button_text = f"✓ {option}" if is_selected else option
                                
                                if st.button(
                                    button_text,
                                    key=f"q_{i}_option_{idx}",
                                    use_container_width=True,
                                    type=button_type
                                ):
                                    # Toggle selection
                                    if option in current_selections:
                                        current_selections.remove(option)
                                    else:
                                        current_selections.append(option)
                                    st.session_state.question_answers[question.question] = current_selections
                                    st.rerun()  # Force rerun to update the display
                        
                        # Show selected answers
                        if current_selections:
                            # if len(current_selections) == 1:
                            #     st.success(f"✓ Selected: {current_selections[0]}")
                            # else:
                            #     st.success(f"✓ Selected: {', '.join(current_selections)}")
                            user_answers[question.question] = current_selections
                    else:
                        # Text input question
                        answer = st.text_input(
                            f"Your answer:",
                            key=f"q_text_{i}",
                            label_visibility="collapsed"
                        )
                        if answer:
                            user_answers[question.question] = [answer]
                    
                    if question.reasoning:
                        st.caption(f"💡 {question.reasoning}")
              # Generate final answer button
            if st.button("📝 Generate Final Answer", disabled=len(user_answers) == 0):
                with st.spinner("Generating comprehensive answer based on your responses..."):
                    try:
                        final_result = run_async(
                            st.session_state.rag_system.finalize_response(
                                st.session_state.analysis_result, 
                                user_answers
                            )
                        )
                        
                        # Debug logging
                        logger.info(f"Final result keys: {final_result.keys()}")
                        logger.info(f"Final answer length: {len(final_result.get('final_answer', ''))}")                        # Display the answer
                        st.markdown("### 📋 Research-Based Answer")
                        
                        if "final_answer" in final_result and final_result["final_answer"]:
                            # Display answer in styled box
                            st.markdown(
                                f'<div class="answer-box">{final_result["final_answer"]}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.error("No answer was generated. Please try again.")
                            logger.error(f"Empty or missing final_answer in result: {final_result}")
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'type': 'answer',
                            'content': final_result["final_answer"],
                            'improved_query': final_result['improved_query'],
                            'sources': final_result['final_results'][:5]  # Top 5 sources
                        })
                          # Clear current questions
                        st.session_state.current_questions = []
                        st.session_state.question_answers = {}
                        
                    except Exception as e:
                        st.error(f"Error generating final answer: {str(e)}")
                        logger.error(f"Answer generation error: {e}")        # Direct answer option (without questions)
        
        elif st.session_state.analysis_result and not st.session_state.current_questions:
            st.info("No clarifying questions needed. You can get a direct answer!")
            
            if st.button("📝 Get Direct Answer"):
                with st.spinner("Generating answer..."):
                    try:
                        answer_result = run_async(
                            st.session_state.rag_system.generate_direct_answer(
                                st.session_state.analysis_result['original_query'],
                                top_k
                            )
                        )                        
                        st.markdown("### 📋 Research-Based Answer")
                        st.success(answer_result["answer"])
                        
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'type': 'direct_answer',
                            'content': answer_result["answer"],
                            'sources': answer_result.get('results', [])
                        })
                        
                    except Exception as e:
                        st.error(f"Error generating direct answer: {str(e)}")
                        logger.error(f"Direct answer error: {e}")
    
    with col2:
        st.header("📚 Source Papers")
        
        # Display source papers if available
        if st.session_state.analysis_result:
            initial_results = st.session_state.analysis_result.get('initial_results', [])
            
            if initial_results:
                st.subheader("🔍 Retrieved Research")
                
                # Group by paper
                papers_data = {}
                for doc in initial_results[:10]:  # Show top 10
                    metadata = doc.get('metadata', {})
                    paper_id = metadata.get('paper_id', 'unknown')
                    
                    if paper_id not in papers_data:
                        papers_data[paper_id] = {
                            'title': doc.get('title', 'Unknown Title'),
                            'authors': doc.get('authors', []),
                            'chunks': []
                        }
                    papers_data[paper_id]['chunks'].append({
                        'text': doc.get('document', '')[:200] + "...",
                        'section': metadata.get('section', 'unknown')
                    })
                
                # Display papers
                for paper_id, paper_data in papers_data.items():
                    # Use Streamlit's built-in info box for guaranteed visibility
                    st.info(f"📄 **{paper_data['title']}**")
                    
                    # Show relevant chunks
                    for chunk in paper_data['chunks'][:2]:  # Show first 2 chunks per paper
                        st.caption(f"*{chunk['section']}*: {chunk['text']}")
                        
        # Conversation history
        st.subheader("💭 Conversation History")
        
        if st.session_state.conversation_history:
            for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
                if item['type'] == 'query':
                    st.markdown(f"**Q{len(st.session_state.conversation_history)-i}:** {item['content']}")
                elif item['type'] in ['answer', 'direct_answer']:
                    content_preview = item['content'][:150] + "..." if len(item['content']) > 150 else item['content']
                    st.markdown(f"**A{len(st.session_state.conversation_history)-i}:** {content_preview}")
                    if 'improved_query' in item:
                        st.caption(f"Refined query: {item['improved_query']}")
                st.write("")  # Small spacing instead of hr
        else:
            st.info("No conversation history yet. Ask a question to get started!")

if __name__ == "__main__":
    main()
