import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliDocs",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 IntelliDocs")

st.subheader(
    "AI-Powered Document Intelligence System"
)

st.write(
    "Upload a PDF and ask questions about its content using RAG."
)

# --------------------------------------------------
# CHECK API KEY
# --------------------------------------------------

if not os.getenv("GOOGLE_API_KEY"):

    st.error(
        "Google API key not found. Please check your .env file."
    )

    st.stop()

# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)

# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # -----------------------------
    # 1. Extract PDF Text
    # -----------------------------

    pdf_reader = PdfReader(
        uploaded_file
    )

    extracted_text = ""

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:

            extracted_text += text + "\n"

    st.info(
        f"📄 Number of pages: {len(pdf_reader.pages)}"
    )

    # --------------------------------------------------
    # CHECK EXTRACTED TEXT
    # --------------------------------------------------

    if extracted_text.strip():

        # -----------------------------
        # 2. Split Text into Chunks
        # -----------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = text_splitter.split_text(
            extracted_text
        )

        st.success(
            f"✂️ Document split into {len(chunks)} chunks."
        )

        # -----------------------------
        # 3. Create Embeddings
        # -----------------------------

        with st.spinner(
            "Creating document embeddings..."
        ):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        st.success(
            "🧠 HuggingFace embeddings created successfully!"
        )

        # -----------------------------
        # 4. Create FAISS Database
        # -----------------------------

        with st.spinner(
            "Building FAISS vector database..."
        ):

            vectorstore = FAISS.from_texts(
                chunks,
                embedding=embeddings
            )

        st.success(
            "🗃️ FAISS vector database created successfully!"
        )

        # -----------------------------
        # 5. Initialize Gemini
        # -----------------------------

        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.2
        )

        # -----------------------------
        # 6. Ask Questions
        # -----------------------------

        st.subheader(
            "💬 Ask Questions About Your PDF"
        )

        query = st.text_input(
            "Enter your question:"
        )

        if query:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                # -----------------------------
                # Retrieve Relevant Chunks
                # -----------------------------

                results = vectorstore.similarity_search(
                    query,
                    k=3
                )

                # -----------------------------
                # Combine Retrieved Context
                # -----------------------------

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in results
                    ]
                )

                # -----------------------------
                # Create RAG Prompt
                # -----------------------------

                prompt = f"""
You are an intelligent document assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not available in the context,
say that the information is not available in the document.

Document Context:
{context}

User Question:
{query}

Answer:
"""

                # -----------------------------
                # Generate Answer with Gemini
                # -----------------------------

                response = llm.invoke(
                    prompt
                )

                # -----------------------------
                # Get Answer
                # -----------------------------

                answer = response.content

                # Handle Gemini structured response
                if isinstance(answer, list):

                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                        and "text" in item
                    )

                # -----------------------------
                # Display Answer
                # -----------------------------

                st.subheader(
                    "🤖 IntelliDocs Answer"
                )

                st.write(
                    answer
                )

                # -----------------------------
                # Show Retrieved Context
                # -----------------------------

                with st.expander(
                    "📚 View Retrieved Document Context"
                ):

                    for i, result in enumerate(
                        results
                    ):

                        st.write(
                            f"**Retrieved Section {i + 1}:**"
                        )

                        st.write(
                            result.page_content
                        )

    else:

        st.warning(
            "No text could be extracted from this PDF."
        )