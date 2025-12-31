# rag_pipeline.py
# The main brain - handles all answers including images!

import streamlit as st  # NEW: For secrets in cloud
import re
import time
from vector_store import create_vector_store, retrieve_chunks
from huggingface_hub import InferenceClient

# NEW: Get the API key from Streamlit secrets (safe for cloud)
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

    # === Page Count (PDF + DOCX) ===
    if "page" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "pages"]):
        if "page_count" in METADATA:
            return f"**The document has {METADATA['page_count']} pages.**"

    # === Row Count (CSV/XLSX) ===
    if "row" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "rows"]):
        if "row_count" in METADATA:
            return f"**The spreadsheet has {METADATA['row_count']} data rows (excluding header).**"

    # === Column Count (CSV/XLSX) ===
    if "column" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "columns"]):
        if "column_count" in METADATA:
            return f"**The spreadsheet has {METADATA['column_count']} columns.**"

    # === Slide Count (PPTX) ===
    if "slide" in query_lower and any(w in query_lower for w in ["how many", "number", "count", "slides"]):
        if "slide_count" in METADATA:
            return f"**The presentation has {METADATA['slide_count']} slides.**"

    # === Word and Letter Counts ===
    has_count_keywords = any(w in query_lower for w in ["count", "number", "how many", "total", "together"])
    if "word" in query_lower or "letter" in query_lower:
        words = len(re.findall(r'\b\w+\b', FULL_TEXT))
        letters = len(re.findall(r'[a-zA-Z]', FULL_TEXT))
        if "word" in query_lower and "letter" in query_lower:
            return f"**Document Statistics:**\n- Words: {words}\n- Letters (alphabetic characters): {letters}"
        elif "word" in query_lower and has_count_keywords:
            return f"**Total words in the document: {words}**"
        elif "letter" in query_lower and has_count_keywords:
            return f"**Total letters (alphabetic characters): {letters}**"

    # === Timesheet Status Count ===
    status_keywords = ["present", "holiday", "leave", "leaves", "absent"]
    mentioned_statuses = [s for s in status_keywords if s in query_lower]
    if any(word in query_lower for word in ["count", "number", "how many", "total"]) and mentioned_statuses:
        try:
            lines = [line.strip() for line in FULL_TEXT.splitlines() if line.strip()]
            status_index = None
            counts = {"Present": 0, "Holiday": 0, "Leave": 0, "Absent": 0}
            for line in lines:
                cells = [c.strip() for c in re.split(r'\s{2,}', line) if c.strip()]
                if not cells:
                    continue
                if status_index is None:
                    try:
                        status_index = cells.index("Status")
                    except ValueError:
                        continue
                else:
                    if len(cells) > status_index:
                        status = cells[status_index]
                        if status in counts:
                            counts[status] += 1
            display_map = {"present": "Present", "holiday": "Holiday", "leaves": "Leave", "leave": "Leave", "absent": "Absent"}
            requested = set(display_map.get(kw, kw.capitalize()) for kw in mentioned_statuses)
            parts = [f"{disp}: {cnt}" for disp, cnt in counts.items() if disp in requested and cnt > 0]
            if parts:
                return "**Attendance Status Count:**\n" + "\n".join(parts)
            else:
                return f"No records found for the requested status(es): {', '.join(requested)}."
        except Exception as e:
            return f"Error analyzing status: {str(e)}"

    # === Special Handling for Images (JPG, JPEG, PNG) ===
    if METADATA.get("is_image", False):
        prompt = f"""You are an expert at analyzing text extracted from images using OCR.
The text below was extracted from the uploaded image. Use it to answer the question as accurately and precisely as possible.

Extracted Text from Image:
{FULL_TEXT}

Question: {query}

Answer:"""

        try:
            response = client.chat_completion(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No clear answer could be determined from the extracted text."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    # === General RAG Query with Retry ===
    if VECTOR_INDEX is None or VECTOR_CHUNKS is None:
        return "Document not indexed yet. Please wait or re-upload."

    retrieved_chunks = retrieve_chunks(VECTOR_INDEX, VECTOR_CHUNKS, query, top_k=15)
    if not retrieved_chunks:
        return "No relevant content found."

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""You are a precise and professional enterprise document assistant.
Answer the question using ONLY the provided context.
Be concise and directly relevant to the question.
Do not add external knowledge, opinions, or assumptions.
Do not hallucinate details not present in the context.

If the answer cannot be clearly and accurately determined from the context, respond exactly with:
"Answer not found in the document."

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
                temperature=0.1,
                stop=["</s>"]
            )
            answer = response.choices[0].message.content.strip()
            return answer if answer else "No response generated."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
            else:
                return "Temporary API connection issue. Please try again in a moment."

    return "Failed to get response after multiple attempts."