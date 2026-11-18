
import gc
import json
import re
import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import faiss
import numpy as np
import pymupdf
import pytesseract
import streamlit as st
import torch

from PIL import Image
from sentence_transformers import SentenceTransformer

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable
)

from rag import generate_answer, generate_suggested_questions


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

torch.set_num_threads(2)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "studymate.db"
STORAGE_DIR = BASE_DIR / "document_storage"


# --------------------------------------------------
# 2. DATABASE FUNCTIONS
# --------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id)
                ON DELETE CASCADE
            )
        """)

        connection.commit()

    finally:
        connection.close()


def create_chat():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO chats (title)
            VALUES (?)
        """, ("New Chat",))

        chat_id = cursor.lastrowid
        connection.commit()

        return chat_id

    finally:
        connection.close()


def get_all_chats():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, title, updated_at
            FROM chats
            ORDER BY updated_at DESC, id DESC
        """)

        return cursor.fetchall()

    finally:
        connection.close()


def format_chat_time(value):
    try:
        parsed_time = datetime.fromisoformat(value)
        return parsed_time.strftime("%d %b %Y · %I:%M %p")

    except (ValueError, TypeError):
        return str(value)


def load_chat_messages(chat_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT role, content, sources_json
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
        """, (chat_id,))

        rows = cursor.fetchall()
        messages = []

        for role, content, sources_json in rows:
            try:
                sources = json.loads(sources_json or "[]")
            except json.JSONDecodeError:
                sources = []

            messages.append({
                "role": role,
                "content": content,
                "sources": sources
            })

        return messages

    finally:
        connection.close()


def save_chat_message(chat_id, role, content, sources=None):
    if sources is None:
        sources = []

    sources_json = json.dumps(
        sources,
        ensure_ascii=False
    )

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO messages (
                chat_id,
                role,
                content,
                sources_json
            )
            VALUES (?, ?, ?, ?)
        """, (chat_id, role, content, sources_json))

        if role == "user":
            cursor.execute("""
                SELECT title
                FROM chats
                WHERE id = ?
            """, (chat_id,))

            result = cursor.fetchone()

            if result and result[0] == "New Chat":
                title = content.strip().replace("\n", " ")

                if len(title) > 45:
                    title = title[:45] + "..."

                cursor.execute("""
                    UPDATE chats
                    SET title = ?
                    WHERE id = ?
                """, (title, chat_id))

        cursor.execute("""
            UPDATE chats
            SET updated_at = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(timespec="seconds"),
            chat_id
        ))

        connection.commit()

    finally:
        connection.close()


initialize_database()


# --------------------------------------------------
# 3. DOCUMENT PERSISTENCE
# --------------------------------------------------

def get_chat_storage_dir(chat_id):
    return STORAGE_DIR / f"chat_{chat_id}"


def reset_document_session():
    st.session_state.document_chunks = None
    st.session_state.index = None
    st.session_state.processed_files = []
    st.session_state.total_pages = 0
    st.session_state.ocr_pages = 0
    st.session_state.suggested_questions = []


def save_document_bundle(
    chat_id,
    uploaded_files,
    document_chunks,
    index,
    total_pages,
    ocr_pages,
    suggested_questions
):
    """
    Save original PDFs, extracted chunks, FAISS index
    and document metadata for a particular chat.
    """

    chat_dir = get_chat_storage_dir(chat_id)
    originals_dir = chat_dir / "originals"

    originals_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for file_number, uploaded_file in enumerate(uploaded_files):
        original_name = Path(uploaded_file.name).name

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]",
            "_",
            original_name
        )

        stored_name = f"{file_number}_{safe_name}"

        file_path = originals_dir / stored_name

        file_path.write_bytes(uploaded_file.getvalue())

        saved_files.append({
            "original_name": original_name,
            "stored_name": stored_name
        })

    chunks_path = chat_dir / "chunks.json"
    index_path = chat_dir / "index.faiss"
    metadata_path = chat_dir / "metadata.json"

    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(
            document_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    faiss.write_index(index, str(index_path))

    metadata = {
        "chat_id": chat_id,
        "saved_files": saved_files,
        "processed_files": [
            file["original_name"] for file in saved_files
        ],
        "total_pages": total_pages,
        "ocr_pages": ocr_pages,
        "suggested_questions": suggested_questions,
        "chunk_count": len(document_chunks),
        "embedding_dimension": index.d,
        "saved_at": datetime.now().isoformat(timespec="seconds")
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_document_bundle(chat_id):
    """
    Restore a chat's document chunks and FAISS index.
    Returns True if the saved bundle is available.
    """

    chat_dir = get_chat_storage_dir(chat_id)

    chunks_path = chat_dir / "chunks.json"
    index_path = chat_dir / "index.faiss"
    metadata_path = chat_dir / "metadata.json"

    if not (
        chunks_path.exists()
        and index_path.exists()
        and metadata_path.exists()
    ):
        return False

    try:
        with open(chunks_path, "r", encoding="utf-8") as file:
            document_chunks = json.load(file)

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        index = faiss.read_index(str(index_path))

        if not document_chunks or index.ntotal == 0:
            return False

        if len(document_chunks) != index.ntotal:
            st.warning(
                "Saved document data is incomplete. "
                "Please process the PDFs again."
            )
            return False

        st.session_state.document_chunks = document_chunks
        st.session_state.index = index

        st.session_state.processed_files = metadata.get(
            "processed_files", []
        )

        st.session_state.total_pages = metadata.get(
            "total_pages", 0
        )

        st.session_state.ocr_pages = metadata.get(
            "ocr_pages", 0
        )

        st.session_state.suggested_questions = metadata.get(
            "suggested_questions", []
        )

        return True

    except Exception as error:
        st.warning(
            f"Could not restore saved documents: {error}"
        )
        return False


# --------------------------------------------------
# 4. SESSION STATE
# --------------------------------------------------

defaults = {
    "messages": [],
    "document_chunks": None,
    "index": None,
    "processed_files": [],
    "total_pages": 0,
    "ocr_pages": 0,
    "suggested_questions": [],
    "pending_question": None,
    "active_chat_id": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


if st.session_state.active_chat_id is None:
    existing_chats = get_all_chats()

    if existing_chats:
        chat_id = existing_chats[0][0]
    else:
        chat_id = create_chat()

    st.session_state.active_chat_id = chat_id

    st.session_state.messages = load_chat_messages(chat_id)

    load_document_bundle(chat_id)


# --------------------------------------------------
# 5. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.2);
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #888888;
        font-size: 1rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .welcome-subtitle {
        color: #888888;
        font-size: 0.95rem;
    }

    .feature-title {
        font-size: 1.05rem;
        font-weight: 650;
    }

    .footer {
        text-align: center;
        color: #888888;
        font-size: 0.8rem;
        padding-top: 35px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 6. LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading StudyMate AI..."):
    model = load_embedding_model()


# --------------------------------------------------
# 7. PDF EXTRACTION WITH OCR
# --------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    pdf_document = pymupdf.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages_data = []
    total_pages = len(pdf_document)
    ocr_pages = 0

    try:
        for page_number, page in enumerate(
            pdf_document,
            start=1
        ):
            text = page.get_text("text").strip()

            if not text:
                pix = page.get_pixmap(
                    dpi=200,
                    alpha=False
                )

                image_bytes = pix.tobytes("png")
                image = Image.open(BytesIO(image_bytes))

                try:
                    text = pytesseract.image_to_string(
                        image
                    ).strip()

                finally:
                    image.close()
                    del image
                    del pix

                if text:
                    ocr_pages += 1

            if text:
                pages_data.append({
                    "document_name": uploaded_file.name,
                    "page_number": page_number,
                    "text": text
                })

    finally:
        pdf_document.close()

    gc.collect()

    return pages_data, total_pages, ocr_pages


# --------------------------------------------------
# 8. TEXT CHUNKING
# --------------------------------------------------

def create_chunks(
    pages_data,
    chunk_size=150,
    overlap=30
):
    document_chunks = []

    for page in pages_data:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk_text = " ".join(words[start:end])

            if chunk_text.strip():
                document_chunks.append({
                    "document_name": page["document_name"],
                    "page_number": page["page_number"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    return document_chunks


# --------------------------------------------------
# 9. FOLLOW-UP QUESTION DETECTION
# --------------------------------------------------

def is_follow_up(user_query):
    query = user_query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)

    follow_up_patterns = [
        r"\bit\b",
        r"\bits\b",
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bthem\b",
        r"\btheir\b",
        r"\bformer\b",
        r"\blatter\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bgive an example\b",
        r"\bexplain more\b",
        r"\belaborate\b",
        r"\btell me more\b",
        r"\bcontinue\b",
        r"\bsame\b",
        r"\bmore details\b",
        r"\bmore about\b",
        r"\bcompare them\b",
        r"\bdifference between them\b"
    ]

    if any(
        re.search(pattern, query)
        for pattern in follow_up_patterns
    ):
        return True

    short_follow_ups = [
        "why",
        "how",
        "why so",
        "how so",
        "example",
        "another example",
        "explain",
        "elaborate",
        "continue",
        "and then",
        "what next"
    ]

    return query in short_follow_ups


def build_retrieval_query(user_query, messages):
    query = user_query.strip()

    if not messages:
        return query

    if not is_follow_up(query):
        return query

    previous_questions = [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]

    if not previous_questions:
        return query

    previous_question = previous_questions[-1]

    return (
        f"Previous topic/question: {previous_question}\n"
        f"Current follow-up question: {query}"
    )

# --------------------------------------------------
# 10. MMR RETRIEVAL WITH RELEVANCE FILTERING
# --------------------------------------------------

RELEVANCE_THRESHOLD = 0.35


def retrieve_chunks(
    query,
    model,
    index,
    document_chunks,
    top_k=5,
    candidate_k=12,
    diversity=0.75
):
    if index is None or index.ntotal == 0:
        return []

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(1, -1)

    candidate_count = min(candidate_k, index.ntotal)

    distances, indices = index.search(
        query_embedding,
        candidate_count
    )

    candidates = []

    for position, chunk_index in enumerate(indices[0]):

        if chunk_index == -1:
            continue

        similarity_score = float(distances[0][position])

        # Remove chunks below the relevance threshold.
        if similarity_score < RELEVANCE_THRESHOLD:
            continue

        chunk_index = int(chunk_index)

        chunk_vector = index.reconstruct(chunk_index)

        candidates.append({
            "index": chunk_index,
            "vector": np.asarray(
                chunk_vector,
                dtype=np.float32
            ),
            "similarity_score": similarity_score
        })

    del query_embedding
    gc.collect()

    # No sufficiently relevant chunks found.
    if not candidates:
        return []

    # Apply Maximum Marginal Relevance (MMR).
    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:

        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:

            relevance = candidate["similarity_score"]

            if selected:

                redundancy_scores = [
                    float(
                        np.dot(
                            candidate["vector"],
                            chosen["vector"]
                        )
                    )
                    for chosen in selected
                ]

                redundancy = max(redundancy_scores)

            else:
                redundancy = 0.0

            mmr_score = (
                diversity * relevance
                - (1 - diversity) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    # Prepare final retrieved chunks.
    retrieved_chunks = []

    for candidate in selected:

        chunk = document_chunks[candidate["index"]]

        retrieved_chunks.append({
            **chunk,
            "similarity_score": candidate["similarity_score"]
        })

    return retrieved_chunks

# --------------------------------------------------
# 11. CALLBACKS
# --------------------------------------------------

def select_suggested_question(question):
    st.session_state.pending_question = question


def open_chat(chat_id):
    st.session_state.active_chat_id = chat_id

    st.session_state.messages = load_chat_messages(
        chat_id
    )

    st.session_state.pending_question = None

    reset_document_session()
    load_document_bundle(chat_id)

    st.rerun()


# --------------------------------------------------
# 12. CHAT EXPORT FUNCTIONS
# --------------------------------------------------

def create_chat_txt(messages):
    lines = [
        "STUDYMATE AI - CHAT EXPORT",
        "=" * 55,
        ""
    ]

    for message in messages:
        role = message["role"].upper()
        content = message["content"]

        lines.append(f"{role}:")
        lines.append(content)
        lines.append("")

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            lines.append("SOURCES:")

            for source_number, source in enumerate(
                message["sources"],
                start=1
            ):
                lines.append(
                    f"{source_number}. "
                    f"{source['document_name']} "
                    f"- Page {source['page_number']}"
                )

                lines.append(
                    f"   Retrieved text: {source['text']}"
                )

                lines.append("")

        lines.append("-" * 55)
        lines.append("")

    lines.append("Generated by StudyMate AI")

    return "\n".join(lines).encode("utf-8")


def create_chat_pdf(messages):
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "StudyMateTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#263B5E"),
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "StudyMateSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20
    )

    user_style = ParagraphStyle(
        "UserMessage",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#245A9B"),
        spaceBefore=10,
        spaceAfter=5
    )

    assistant_style = ParagraphStyle(
        "AssistantMessage",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#222222"),
        spaceBefore=10,
        spaceAfter=5
    )

    source_heading_style = ParagraphStyle(
        "SourceHeading",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#287A50"),
        spaceBefore=7,
        spaceAfter=3
    )

    source_text_style = ParagraphStyle(
        "SourceText",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
        leftIndent=12,
        spaceAfter=5
    )

    story = []

    story.append(
        Paragraph("StudyMate AI", title_style)
    )

    story.append(
        Paragraph(
            "Your academic chat history",
            subtitle_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.lightgrey
        )
    )

    story.append(Spacer(1, 12))

    def prepare_pdf_text(value):
        text = str(value)

        text = text.encode(
            "latin-1",
            errors="replace"
        ).decode("latin-1")

        return escape(text).replace("\n", "<br/>")

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            heading = "YOU"
            message_style = user_style
        else:
            heading = "STUDYMATE AI"
            message_style = assistant_style

        story.append(
            Paragraph(
                f"<b>{heading}</b>",
                message_style
            )
        )

        story.append(
            Paragraph(
                prepare_pdf_text(content),
                message_style
            )
        )

        if role == "assistant" and message.get("sources"):
            story.append(
                Paragraph(
                    "<b>Sources used:</b>",
                    source_heading_style
                )
            )

            for source_number, source in enumerate(
                message["sources"],
                start=1
            ):
                source_heading = (
                    f"{source_number}. "
                    f"{source['document_name']} "
                    f"- Page {source['page_number']}"
                )

                story.append(
                    Paragraph(
                        prepare_pdf_text(source_heading),
                        source_heading_style
                    )
                )

                story.append(
                    Paragraph(
                        prepare_pdf_text(source["text"]),
                        source_text_style
                    )
                )

        story.append(Spacer(1, 8))

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.lightgrey
            )
        )

    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Generated by StudyMate AI",
            subtitle_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# --------------------------------------------------
# 13. SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📚 StudyMate AI")
    st.caption("Your personal AI study assistant")

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        new_chat_id = create_chat()

        st.session_state.active_chat_id = new_chat_id
        st.session_state.messages = []
        st.session_state.pending_question = None

        reset_document_session()

        st.rerun()

    st.markdown("### 🕘 Chat History")

    all_chats = get_all_chats()

    if all_chats:
        for chat_id, title, updated_at in all_chats:
            is_active = (
                chat_id == st.session_state.active_chat_id
            )

            button_label = (
                f"📌 {title}" if is_active
                else f"💬 {title}"
            )

            if st.button(
                button_label,
                key=f"history_chat_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                open_chat(chat_id)

            st.caption(
                f"🕒 {format_chat_time(updated_at)}"
            )

    else:
        st.caption("No saved conversations yet.")

    st.divider()

    st.markdown("### 📄 Your Documents")

    uploaded_files = st.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.caption(
            f"{len(uploaded_files)} PDF(s) selected"
        )

        for uploaded_file in uploaded_files:
            st.write(f"📄 {uploaded_file.name}")

    process_button = st.button(
        "⚡ Process Documents",
        use_container_width=True,
        disabled=not uploaded_files
    )

    st.divider()

    st.markdown("### 📊 Document Status")

    if st.session_state.document_chunks is not None:

        st.success("Documents ready")

        st.metric("Total Pages", st.session_state.total_pages)
        st.metric("OCR Pages", st.session_state.ocr_pages)

        st.metric(
            "Text Chunks",
            len(st.session_state.document_chunks)
        )

        st.metric(
            "Embedding Dimensions",
            st.session_state.index.d
        )

        st.caption("Processed documents:")

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    else:
        st.info(
            "Upload and process PDFs to start chatting."
        )

    st.divider()

    st.markdown("### 💬 Chat Controls")

    if st.button(
        "🗑️ Clear Current Chat",
        use_container_width=True,
        disabled=not st.session_state.messages
    ):
        connection = get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("""
                DELETE FROM messages
                WHERE chat_id = ?
            """, (st.session_state.active_chat_id,))

            cursor.execute("""
                UPDATE chats
                SET title = 'New Chat'
                WHERE id = ?
            """, (st.session_state.active_chat_id,))

            connection.commit()

        finally:
            connection.close()

        st.session_state.messages = []
        st.session_state.pending_question = None

        st.rerun()

    st.divider()

    st.markdown("### 📥 Export Chat")

    has_messages = bool(st.session_state.messages)

    if has_messages:
        txt_data = create_chat_txt(
            st.session_state.messages
        )

        pdf_data = create_chat_pdf(
            st.session_state.messages
        )

    else:
        txt_data = b""
        pdf_data = b""

    st.download_button(
        label="📄 Download as TXT",
        data=txt_data,
        file_name="StudyMate_Chat.txt",
        mime="text/plain",
        use_container_width=True,
        disabled=not has_messages
    )

    st.download_button(
        label="📕 Download as PDF",
        data=pdf_data,
        file_name="StudyMate_Chat.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=not has_messages
    )

    st.caption("Your conversations are saved locally.")


# --------------------------------------------------
# 14. PROCESS DOCUMENTS
# --------------------------------------------------

if process_button:

    reset_document_session()

    st.session_state.messages = []
    st.session_state.pending_question = None

    new_chat_id = create_chat()
    st.session_state.active_chat_id = new_chat_id

    gc.collect()

    all_pages = []
    total_pages = 0
    total_ocr_pages = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.info(
                f"📖 Reading {uploaded_file.name}..."
            )

            pages_data, page_count, ocr_count = (
                extract_text_from_pdf(uploaded_file)
            )

            all_pages.extend(pages_data)

            total_pages += page_count
            total_ocr_pages += ocr_count

            progress_bar.progress(
                (i + 1) / len(uploaded_files)
            )

        status_text.info("✂️ Creating text chunks...")

        document_chunks = create_chunks(all_pages)

        if not document_chunks:
            status_text.empty()
            progress_bar.empty()

            st.error(
                "No text could be extracted from the PDFs."
            )

            st.stop()

        del all_pages
        gc.collect()

        status_text.info("🧠 Generating embeddings...")

        chunk_texts = [
            chunk["text"]
            for chunk in document_chunks
        ]

        embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=False
        )

        embedding_matrix = np.asarray(
            embeddings,
            dtype=np.float32
        )

        status_text.info("🔍 Building search index...")

        dimension = embedding_matrix.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embedding_matrix)

        del chunk_texts
        del embeddings
        del embedding_matrix

        gc.collect()

        st.session_state.document_chunks = document_chunks
        st.session_state.index = index

        st.session_state.total_pages = total_pages
        st.session_state.ocr_pages = total_ocr_pages

        st.session_state.processed_files = [
            file.name for file in uploaded_files
        ]

        status_text.info(
            "💡 Generating questions from your PDFs..."
        )

        try:
            suggested_questions = generate_suggested_questions(
                document_chunks,
                count=4
            )

            st.session_state.suggested_questions = (
                suggested_questions
            )

        except Exception as question_error:
            st.session_state.suggested_questions = []

            st.warning(
                "Documents were processed, but suggested "
                f"questions could not be generated: {question_error}"
            )

        # Save the PDFs, chunks, FAISS index and metadata.
        status_text.info("💾 Saving documents for future sessions...")

        save_document_bundle(
            chat_id=st.session_state.active_chat_id,
            uploaded_files=uploaded_files,
            document_chunks=document_chunks,
            index=index,
            total_pages=total_pages,
            ocr_pages=total_ocr_pages,
            suggested_questions=st.session_state.suggested_questions
        )

        status_text.empty()
        progress_bar.empty()

        st.success(
            "Documents processed and saved successfully!"
        )

        st.rerun()

    except Exception as error:
        status_text.empty()
        progress_bar.empty()

        gc.collect()

        st.error(
            f"Document processing failed: {error}"
        )


# --------------------------------------------------
# 15. MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">StudyMate AI 📚</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your personal AI-powered academic PDF assistant.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# 16. WELCOME SCREEN
# --------------------------------------------------

if (
    st.session_state.document_chunks is None
    and not st.session_state.messages
):

    with st.container(border=True):
        st.markdown(
            '<div class="welcome-title">'
            '👋 Welcome to StudyMate AI'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="welcome-subtitle">'
            'Upload your academic PDFs from the sidebar '
            'and start asking questions.'
            '<br><br>'
            'Study smarter with answers grounded in your documents.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("### ✨ What you can do")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 📄")
            st.markdown(
                '<div class="feature-title">'
                'PDF Understanding</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Extract information from text-based "
                "and scanned PDF documents."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### 🧠")
            st.markdown(
                '<div class="feature-title">'
                'AI Answers</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "Ask questions and receive answers "
                "based on your study materials."
            )

    with col3:
        with st.container(border=True):
            st.markdown("### 📚")
            st.markdown(
                '<div class="feature-title">'
                'Source References</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "View the document pages used "
                "to generate each answer."
            )


# --------------------------------------------------
# 17. CHAT INTERFACE
# --------------------------------------------------

else:

    st.markdown("### 💬 Chat with StudyMate AI")

    if st.session_state.document_chunks is not None:

        st.caption(
            "Ask questions, request explanations, "
            "or continue with follow-up questions."
        )

        if (
            not st.session_state.messages
            and st.session_state.suggested_questions
        ):

            st.markdown("#### 💡 Suggested questions")

            st.caption(
                "These questions were generated from your uploaded PDFs."
            )

            question_columns = st.columns(2)

            for i, question in enumerate(
                st.session_state.suggested_questions
            ):

                with question_columns[i % 2]:
                    st.button(
                        question,
                        key=f"suggested_question_{i}",
                        use_container_width=True,
                        on_click=select_suggested_question,
                        args=(question,)
                    )

    else:
        st.info(
            "You are viewing a saved conversation without "
            "available document data. Upload and process the "
            "original PDF again if its saved document data "
            "cannot be restored."
        )

    # Display saved messages.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 View sources used for this answer"
                ):

                    for source_number, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        with st.container(border=True):
                            st.markdown(
                                f"**{source_number}. "
                                f"{source['document_name']}**"
                            )

                            st.caption(
                                f"Page {source['page_number']} "
                                f"· Similarity: "
                                f"{source.get('similarity_score', 0):.4f}"
                            )

                            st.write(source["text"])

    # Chat input.
    if st.session_state.document_chunks is not None:

        chat_input = st.chat_input(
            "Ask anything about your uploaded PDFs..."
        )

        pending_question = st.session_state.pending_question

        if pending_question:
            st.session_state.pending_question = None

        user_query = chat_input or pending_question

        if user_query and user_query.strip():

            user_query = user_query.strip()

            retrieval_query = build_retrieval_query(
                user_query,
                st.session_state.messages
            )

            st.session_state.messages.append({
                "role": "user",
                "content": user_query,
                "sources": []
            })

            save_chat_message(
                st.session_state.active_chat_id,
                "user",
                user_query
            )

            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):

                try:
                    with st.spinner(
                        "🔍 Searching your documents..."
                    ):

                        retrieved_chunks = retrieve_chunks(
                            retrieval_query,
                            model,
                            st.session_state.index,
                            st.session_state.document_chunks,
                            top_k=5,
                            candidate_k=12,
                            diversity=0.75
                        )

                    if not retrieved_chunks:
                        answer = (
                            "I could not find relevant information "
                            "in the uploaded PDFs for this question."
                        )

                    else:
                        recent_history = (
                            st.session_state.messages[-7:-1]
                        )

                        with st.spinner(
                            "🤖 StudyMate is preparing your answer..."
                        ):

                            answer = generate_answer(
                                user_query,
                                retrieved_chunks,
                                chat_history=recent_history
                            )

                    st.markdown(answer)

                    if retrieved_chunks:

                        with st.expander(
                            "📚 View sources used for this answer"
                        ):

                            for source_number, source in enumerate(
                                retrieved_chunks,
                                start=1
                            ):

                                with st.container(border=True):
                                    st.markdown(
                                        f"**{source_number}. "
                                        f"{source['document_name']}**"
                                    )

                                    st.caption(
                                        f"Page {source['page_number']} "
                                        f"· Similarity: "
                                        f"{source['similarity_score']:.4f}"
                                    )

                                    st.write(source["text"])

                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": retrieved_chunks
                    }

                    st.session_state.messages.append(
                        assistant_message
                    )

                    save_chat_message(
                        st.session_state.active_chat_id,
                        "assistant",
                        answer,
                        retrieved_chunks
                    )

                except Exception as error:
                    error_message = (
                        f"Could not generate an answer: {error}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "sources": []
                    })

                    save_chat_message(
                        st.session_state.active_chat_id,
                        "assistant",
                        error_message,
                        []
                    )

            st.rerun()


# --------------------------------------------------
# 18. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    Built with Python · Streamlit · FAISS · Sentence Transformers · Gemini · SQLite
    <br>
    StudyMate AI — Learn smarter, not harder.
</div>
""", unsafe_allow_html=True)
