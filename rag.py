import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # loads HF_TOKEN from .env

CHROMA_PATH = "./chroma_db"
PDF_FOLDER  = "./pdfs"

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        # HF_TOKEN auto-picked from environment — never hardcoded
    )

def load_or_create_vectorstore(embedding_model, status_callback=None):
    """
    Loads vectorstore from disk if it exists,
    otherwise builds it from PDFs in ./pdfs folder.
    status_callback: optional fn(str) to stream progress to Streamlit
    """
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        if status_callback:
            status_callback("[+--] Loading existing vectorstore from disk...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
        if status_callback:
            status_callback(f"[=V=] Loaded! {vectorstore._collection.count()} vectors found.")
        return vectorstore

    # Build from scratch
    if status_callback:
        status_callback("[>>>] No vectorstore found. Loading PDFs...")

    all_documents = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_FOLDER}/")

    for pdf_file in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        if status_callback:
            status_callback(f"  Loading: {pdf_file}")
        loader = PyPDFLoader(path)
        all_documents.extend(loader.load())

    if status_callback:
        status_callback(f"Loaded {len(all_documents)} pages. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_documents)

    if status_callback:
        status_callback(f"{len(chunks)} chunks created. Embedding (one-time)...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    if status_callback:
        status_callback(f"[=V=] Vectorstore saved to disk! {vectorstore._collection.count()} vectors.")

    return vectorstore


def build_rag_chain(groq_api_key: str, vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=groq_api_key       # user-provided at runtime, never stored
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful physics tutor. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information on that."

Context:
{context}

Question: {question}

Answer:
""")

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain