# app.py
# Main file for the Enterprise AI Assistant

import streamlit as st
from document_loader import load_document
from rag_pipeline import build_rag_index, get_answer

st.set_page_config(page_title="Enterprise AI Assistant", layout="wide", initial_sidebar_state="collapsed")

# Title and Clear button
col_title, col_clear = st.columns([8, 2])
with col_title:
    st.title("Enterprise AI Assistant")
with col_clear:
    if st.button("Clear Conversation", key="clear_btn", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.greeted = False
        st.rerun()

st.markdown("Upload a document and ask questions to receive accurate, precise answers based on its content.")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# File uploader (jpeg removed as you wanted)
uploaded_file = st.file_uploader(
    "Upload document (PDF, Excel, Word, PPT, TXT, Images)",
    type=["pdf", "docx", "pptx", "csv", "txt", "xlsx", "jpg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.chat_history = []
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.greeted = False
   
    with st.spinner("Loading and indexing document..."):
        document_data = load_document(uploaded_file)
        build_rag_index(document_data)
   
    st.success(f"**{uploaded_file.name}** has been successfully loaded and indexed.")

# Welcome message
if uploaded_file and not st.session_state.greeted:
    welcome_msg = "Your document is ready for analysis. Please feel free to ask any questions about its content."
    st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
    st.session_state.greeted = True

# Chat display
chat_container = st.container(height=600)
with chat_container:
    MAX_MESSAGES = 50
    for message in st.session_state.chat_history[-MAX_MESSAGES:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User question input
user_query = st.chat_input("Ask a question about the document...")
if user_query:
    user_query_lower = user_query.lower().strip()
    words = user_query_lower.split()

    # Strict greeting check
    greeting_words = ["hi", "hello", "hey", "hii", "heyy", "helloo", "good morning", "good afternoon", "good evening"]
    is_pure_greeting = user_query_lower in greeting_words or (
        len(words) <= 3 and any(word in greeting_words for word in words) and 
        all(word in greeting_words + [",", "!"] for word in words if word not in greeting_words)
    )

    # Thanks check
    is_pure_thanks = ("thank" in words or "thanks" in words) and len(words) <= 5 and \
                     not any(q in user_query_lower for q in ["what", "how", "why", "is", "describe"])

    # Goodbye check
    goodbye_words = ["bye", "goodbye", "see you", "tata", "take care"]
    is_pure_goodbye = len(words) <= 4 and any(word in goodbye_words for word in words)

    if is_pure_greeting:
        response = "Greetings. How may I assist you with the document today?"
    elif is_pure_thanks:
        response = "You are welcome. Is there anything else I can help with?"
    elif is_pure_goodbye:
        response = "Goodbye. Have a productive day."
    else:
        with st.spinner("Analyzing document..."):
            response = get_answer(user_query)
        
        if ("not found" not in response.lower() and
            "error" not in response.lower() and
            "please upload" not in response.lower() and
            "no relevant" not in response.lower()):
            response += "\n\nPlease let me know if you require further clarification."

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    if len(st.session_state.chat_history) > MAX_MESSAGES:
        st.session_state.chat_history = st.session_state.chat_history[-MAX_MESSAGES:]

    st.rerun()