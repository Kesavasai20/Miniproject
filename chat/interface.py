"""
Chat Interface
Streamlit chat component with rich responses
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import streamlit as st

import sys
sys.path.append('..')
from ai.rag_engine import RAGEngine
from ai.nl2sql import NL2SQLTranslator

logger = logging.getLogger(__name__)


class ChatInterface:
    """Manages chat interactions and state"""
    
    def __init__(self):
        self.rag_engine = RAGEngine()
        self.nl2sql = NL2SQLTranslator()
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = self._default_suggestions()
    
    def _default_suggestions(self) -> List[str]:
        """Default query suggestions"""
        return [
            "Show me all floats in the Arabian Sea",
            "What's the average temperature at 500m depth?",
            "Display floats with oxygen sensors",
            "Compare Bay of Bengal and Arabian Sea salinity",
            "Find recent anomalies in the data"
        ]
    
    def render(self, on_visualization: Optional[Callable] = None):
        """Render the chat interface"""
        st.markdown("### 💬 FloatChat Assistant")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"], avatar="🌊" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])
                    if msg.get("visualization"):
                        st.plotly_chart(msg["visualization"], use_container_width=True)
        
        # Suggestion chips
        if not st.session_state.chat_history:
            st.markdown("**Try asking:**")
            cols = st.columns(2)
            for i, sugg in enumerate(st.session_state.suggestions[:4]):
                with cols[i % 2]:
                    if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                        self._handle_query(sugg, on_visualization)
                        st.rerun()
        
        # Input
        user_input = st.chat_input("Ask about ARGO ocean data...")
        if user_input:
            self._handle_query(user_input, on_visualization)
            st.rerun()
    
    def _handle_query(self, query: str, on_viz: Optional[Callable] = None):
        """Process user query"""
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        try:
            result = self.rag_engine.query(query)
            response = result.get("response", "I couldn't process that query.")
            intent = result.get("intent", "general")
            
            msg = {
                "role": "assistant",
                "content": response,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            }
            
            # If visualization intent, callback
            if intent == "visualization" and on_viz:
                viz = on_viz(query, result)
                if viz:
                    msg["visualization"] = viz
            
            st.session_state.chat_history.append(msg)
            
            # Update suggestions
            st.session_state.suggestions = self.rag_engine.suggest_queries(query)
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Sorry, I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    def clear_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.session_state.suggestions = self._default_suggestions()
    
    def export_history(self) -> List[Dict]:
        """Export chat history"""
        return st.session_state.chat_history.copy()


def render_chat_sidebar():
    """Render chat in sidebar mode"""
    with st.sidebar:
        st.markdown("### 💬 Chat")
        
        if 'sidebar_chat' not in st.session_state:
            st.session_state.sidebar_chat = []
        
        # Display messages
        for msg in st.session_state.sidebar_chat[-5:]:  # Last 5 messages
            role = "🌊" if msg["role"] == "assistant" else "👤"
            st.markdown(f"{role} {msg['content'][:200]}...")
        
        # Input
        query = st.text_input("Ask...", key="sidebar_query")
        if st.button("Send", key="sidebar_send") and query:
            st.session_state.sidebar_chat.append({"role": "user", "content": query})
            # Process...
            st.rerun()
