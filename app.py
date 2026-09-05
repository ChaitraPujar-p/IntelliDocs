import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="IntelliDocs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.2rem;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .feature-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        opacity: 0.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):

    st.error(
        "Google API key not found. Please configure GOOGLE_API_KEY."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "page_count" not in st.session_state:
    st.session_state.page_count = 0


# ============================================================
# CACHED EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# CACHED GEMINI
# ============================================================

@st.cache_resource(show_spinner=False)
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.2
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 IntelliDocs")

    st.write(
        "AI-powered document intelligence using "
        "Retrieval-Augmented Generation."
    )

    st.divider()

    st.subheader("🧠 Technology Stack")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• LangChain")
    st.write("• HuggingFace")
    st.write("• FAISS")
    st.write("• Google Gemini")

    st.divider()

    st.subheader("⚙️ RAG Pipeline")

    st.write(
        "PDF → Text Extraction → Chunking → "
        "Embeddings → FAISS → Retrieval → Gemini"
    )

    st.divider()

    st.caption(
        "IntelliDocs | AI Document Intelligence"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 IntelliDocs</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Document Intelligence System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a PDF, retrieve relevant information, "
    "and ask questions using Retrieval-Augmented Generation (RAG)."
)


# ============================================================
# UPLOAD
# ============================================================

st.subheader("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF for best results."
)


# ============================================================
# PROCESS DOCUMENT
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    document_id = hashlib.md5(
        file_bytes
    ).hexdigest()


    # Process only new document
    if st.session_state.document_id != document_id:

        st.session_state.vectorstore = None
        st.session_state.document_id = document_id
        st.session_state.document_name = uploaded_file.name


        # ----------------------------------------------------
        # PDF EXTRACTION
        # ----------------------------------------------------

        with st.spinner("📖 Reading document..."):

            try:

                pdf_reader = PdfReader(
                    uploaded_file
                )

                extracted_text = ""

                for page in pdf_reader.pages:

                    text = page.extract_text()

                    if text:

                        extracted_text += text + "\n"

                st.session_state.page_count = len(
                    pdf_reader.pages
                )

            except Exception as e:

                st.error(
                    f"Unable to read PDF: {e}"
                )

                st.stop()


        if not extracted_text.strip():

            st.warning(
                "No text could be extracted from this PDF."
            )

            st.info(
                "This may be a scanned PDF. OCR support "
                "would be required."
            )

            st.stop()


        # ----------------------------------------------------
        # CHUNKING
        # ----------------------------------------------------

        with st.spinner("✂️ Splitting document..."):

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(
                extracted_text
            )

            st.session_state.chunks_count = len(chunks)


        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        with st.spinner(
            "🧠 Loading embedding model..."
        ):

            embeddings = load_embeddings()


        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        with st.spinner(
            "🗃️ Building FAISS vector database..."
        ):

            vectorstore = FAISS.from_texts(
                chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore


        st.success(
            "✅ Document processed successfully!"
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            st.session_state.page_count
        )

    with col2:

        st.metric(
            "Text Chunks",
            st.session_state.chunks_count
        )

    with col3:

        st.metric(
            "Status",
            "Ready"
        )


    st.info(
        f"📄 **{st.session_state.document_name}** "
        "is ready for questions."
    )


    # ========================================================
    # QUESTION SECTION
    # ========================================================

    st.subheader(
        "💬 Ask Questions About Your Document"
    )

    with st.form("question_form"):

        query = st.text_input(
            "Enter your question",
            placeholder=(
                "Example: What are the main objectives?"
            )
        )

        ask_button = st.form_submit_button(
            "🔍 Ask Question"
        )


    # ========================================================
    # ANSWER
    # ========================================================

    if ask_button and query.strip():

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching relevant document sections..."
        ):

            results = (
                st.session_state.vectorstore
                .similarity_search(
                    query,
                    k=3
                )
            )


        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in results
        )


        # ----------------------------------------------------
        # PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are IntelliDocs, an intelligent document assistant.

Answer the user's question using ONLY the document
context provided below.

Rules:

1. Use only the provided document context.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer cannot be found in the context,
   say:

"The information is not available in the document."

5. Give a clear and concise answer.

Document Context:
{context}

User Question:
{query}

Answer:
"""


        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            try:

                llm = load_llm()

                response = llm.invoke(
                    prompt
                )

                answer = response.content

            except Exception as e:

                st.error(
                    f"Gemini error: {e}"
                )

                st.stop()


        # ----------------------------------------------------
        # RESPONSE HANDLING
        # ----------------------------------------------------

        if isinstance(answer, list):

            answer = "\n".join(
                item.get("text", "")
                for item in answer
                if isinstance(item, dict)
                and "text" in item
            )


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        st.subheader(
            "🤖 IntelliDocs Answer"
        )

        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # RETRIEVED CONTEXT
        # ----------------------------------------------------

        with st.expander(
            "📚 View Retrieved Document Context"
        ):

            for i, result in enumerate(results):

                st.markdown(
                    f"### Retrieved Section {i + 1}"
                )

                st.write(
                    result.page_content
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
    Built with Python • LangChain • FAISS • HuggingFace • Google Gemini
    <br>
    IntelliDocs — AI-Powered Document Intelligence
    </div>
    """,
    unsafe_allow_html=True
)