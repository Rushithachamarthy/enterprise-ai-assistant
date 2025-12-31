# rag_pipeline.py
# The main brain - handles all answers including images!

import streamlit as st
import re
import time
from vector_store import create_vector_store, retrieve_chunks
from huggingface_hub import InferenceClient

# Get the API key from Streamlit secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in Streamlit secrets! Please add it in app settings.")

HF_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Global state
VECTOR_INDEX = None
VECTOR_CHUNKS = None
FULL_TEXT = None
METADATA = None

client = InferenceClient(token=HF_API_KEY)

def build_rag_index(document_data):
    global VECTOR_INDEX, VECTOR_CHUNKS, FULL_TEXT, METADATA
    if isinstance(document_data, tuple):
        FULL_TEXT, METADATA = document_data
    else:
        FULL_TEXT = document_data
        METADATA = {}
    VECTOR_INDEX, VECTOR_CHUNKS = create_vector_store(FULL_TEXT)

def get_answer(query):
    global VECTOR_INDEX, VECTOR_CHUNKS, FULL_TEXT, METADATA
    
    if FULL_TEXT is None:
        return "Please upload a document first."

    query_lower = query.lower().strip()

    # === Smart Attendance Counting (Works on Images & Tables) ===
    count_keywords = ["count", "number", "how many", "total"]
    status_keywords = ["present", "p ", "holiday", "leave", "absent", "l ", "h "]
    
    if any(word in query_lower for word in count_keywords) and any(word in query_lower for word in status_keywords):
        # Extract all lines and try to find table-like structure
        lines = [line.strip() for line in FULL_TEXT.splitlines() if line.strip()]
        
        # Possible headers for status column
        possible_headers = ["status", "attendance", "remark", "sts", "att", "day", "present"]
        header_row = None
        status_col_index = -1
        
        for i, line in enumerate(lines[:10]):  # Check first 10 lines for header
            lower_line = line.lower()
            for header in possible_headers:
                if header in lower_line:
                    header_row = i
                    cells = re.split(r'\s{2,}|\t', line)  # Split by multiple spaces or tab
                    for j, cell in enumerate(cells):
                        if header in cell.lower():
                            status_col_index = j
                            break
                    if status_col_index != -1:
                        break
            if status_col_index != -1:
                break
        
        # Normalize status values
        status_map = {
            "p": "Present", "present": "Present", "pres": "Present",
            "a": "Absent", "absent": "Absent",
            "h": "Holiday", "holiday": "Holiday",
            "l": "Leave", "leave": "Leave", "leaves": "Leave"
        }
        
        counts = {"Present": 0, "Absent": 0, "Holiday": 0, "Leave": 0}
        
        start_row = header_row + 1 if header_row is not None else 0
        for line in lines[start_row:]:
            cells = re.split(r'\s{2,}|\t', line)
            if status_col_index < len(cells):
                status = cells[status_col_index].strip().lower()
                normalized = status_map.get(status, None)
                if normalized:
                    counts[normalized] += 1
                elif status in ["p", "present", "h", "holiday", "l", "leave", "a", "absent"]:
                    short_map = {"p": "Present", "a": "Absent", "h": "Holiday", "l": "Leave"}
                    counts[short_map[status]] += 1
        
        # Build response for requested statuses
        requested = [word for word in status_keywords if word.rstrip(" ") in query_lower]
        display_map = {"present": "Present", "p ": "Present", "holiday": "Holiday", "leave": "Leave", "absent": "Absent"}
        requested_names = [display_map.get(k, k.capitalize().rstrip()) for k in requested]
        
        parts = []
        for name in requested_names:
            if counts.get(name, 0) > 0:
                parts.append(f"**{name}**: {counts[name]}")
        
        if parts:
            return "**Attendance Count:**\n" + "\n".join(parts) + "\n\nPlease let me know if you need more details."
        else:
            # Fallback: Let AI count directly from full text
            pass  # We'll use general RAG below

    # === Other Metadata Queries (Page count, etc.) ===
    if "page" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "pages"]):
        if "page_count" in METADATA:
            return f"**The document has {METADATA['page_count']} pages.**"

    if "row" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "rows"]):
        if "row_count" in METADATA:
            return f"**The spreadsheet has {METADATA['row_count']} data rows.**"

    if "column" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "columns"]):
        if "column_count" in METADATA:
            return f"**The spreadsheet has {METADATA['column_count']} columns.**"

    if "slide" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "slides"]):
        if "slide_count" in METADATA:
            return f"**The presentation has {METADATA['slide_count']} slides.**"

    # === Special Handling for Images (Improved Prompt) ===
    if METADATA.get("is_image", False):
        prompt = f"""You are an expert attendance analyzer. Extract and analyze the attendance data from the OCR text below.

Extracted Text:
{FULL_TEXT}

Question: {query}

Provide a clear, accurate answer based only on the text. If counting attendance (Present, Absent, Holiday, Leave, P, A, H, L), give exact numbers.
If the question is about counting specific marks, respond directly with the count.

Answer:"""

        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "Could not analyze the image content."
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # === General RAG for everything else ===
    if VECTOR_INDEX is None or VECTOR_CHUNKS is None:
        return "Document not indexed yet. Please wait or re-upload."

    retrieved_chunks = retrieve_chunks(VECTOR_INDEX, VECTOR_CHUNKS, query, top_k=15)
    if not retrieved_chunks:
        return "No relevant content found in the document."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a precise enterprise document assistant.
Use ONLY the provided context to answer the question.
Be direct and professional.

Context:
{context}

Question: {query}

Answer:"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No clear answer found."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                return "Temporary issue. Please try again."

    return "Failed to generate response."