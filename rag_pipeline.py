# rag_pipeline.py
# Fixed version: Handles ALL images perfectly, no more false attendance triggers

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

    # === PRIORITY 1: Handle Images FIRST (before any other logic) ===
    if METADATA.get("is_image", False):
        prompt = f"""You are an expert document and image analyst. Your job is to accurately read and interpret the content from the extracted OCR text below.

Extracted Text from Image:
{FULL_TEXT}

User Question: {query}

Instructions:
- Answer directly and clearly based only on the visible text in the image.
- If asked for names, list all visible names.
- If asked to count something, give exact counts.
- If it's a table, extract it properly.
- If it's a form, screenshot, or handwritten text, do your best to interpret it.
- Be concise and professional.

Answer:"""

        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.3,
                stop=["</s>", "<|eot_id|>"]
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No clear text could be extracted from the image."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    # === PRIORITY 2: Metadata Queries (pages, rows, slides, etc.) ===
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

    # === PRIORITY 3: Smart Attendance Counting (ONLY for text/tables, not images) ===
    count_keywords = ["count", "number", "how many", "total"]
    status_indicators = ["present", "absent", "holiday", "leave", "p", "a", "h", "l"]

    is_count_query = any(word in query_lower for word in count_keywords)
    is_attendance_query = any(ind in query_lower for ind in status_indicators)

    if is_count_query and is_attendance_query:
        lines = [line.strip() for line in FULL_TEXT.splitlines() if line.strip()]

        possible_headers = ["status", "attendance", "remark", "sts", "att", "day", "present", "absent"]
        header_row = None
        status_col_index = -1

        for i, line in enumerate(lines[:15]):
            lower_line = line.lower()
            cells = re.split(r'\s{2,}|\t', line)
            for j, cell in enumerate(cells):
                if any(h in cell.lower() for h in possible_headers):
                    status_col_index = j
                    header_row = i
                    break
            if status_col_index != -1:
                break

        status_map = {
            "p": "Present", "present": "Present", "pres": "Present",
            "a": "Absent", "absent": "Absent",
            "h": "Holiday", "holiday": "Holiday",
            "l": "Leave", "leave": "Leave"
        }

        counts = {"Present": 0, "Absent": 0, "Holiday": 0, "Leave": 0}
        start_row = header_row + 1 if header_row is not None else 0

        for line in lines[start_row:]:
            cells = re.split(r'\s{2,}|\t', line)
            if status_col_index < len(cells):
                status = cells[status_col_index].strip().lower()
                normalized = status_map.get(status) or status_map.get(status[0] if status else "")
                if normalized:
                    counts[normalized] += 1

        # Extract which statuses user asked for
        asked = [ind for ind in status_indicators if ind in query_lower]
        display_names = {"p": "Present", "a": "Absent", "h": "Holiday", "l": "Leave",
                         "present": "Present", "absent": "Absent", "holiday": "Holiday", "leave": "Leave"}

        parts = []
        for ask in asked:
            name = display_names.get(ask, ask.capitalize())
            if counts.get(name, 0) > 0:
                parts.append(f"**{name}**: {counts[name]}")

        if parts:
            return "**Attendance Summary:**\n" + "\n".join(parts)

        # If no structured table found, fall through to general RAG

    # === FINAL: General RAG for all other queries ===
    if VECTOR_INDEX is None or VECTOR_CHUNKS is None:
        return "Document not indexed yet. Please wait or re-upload."

    retrieved_chunks = retrieve_chunks(VECTOR_INDEX, VECTOR_CHUNKS, query, top_k=15)
    if not retrieved_chunks:
        return "No relevant content found in the document."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a precise enterprise document assistant.
Use ONLY the provided context to answer the question.
Be direct, professional, and accurate.

Context:
{context}

Question: {query}

Answer:"""

    for attempt in range(5):
        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.1
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No clear answer found."
        except Exception:
            if attempt < 4:
                time.sleep(2 ** attempt)
            else:
                return "Temporary issue with the AI model. Please try again."

    return "Failed to generate response."