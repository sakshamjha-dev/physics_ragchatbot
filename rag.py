import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

FAISS_PATH = "./faiss_db"
PDF_FOLDER = "./pdfs"

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def load_or_create_vectorstore(embedding_model, status_callback=None):
    if os.path.exists(FAISS_PATH) and os.listdir(FAISS_PATH):
        if status_callback:
            status_callback("[+--] Loading vectorstore from disk...")
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        if status_callback:
            status_callback("[=V=] Loaded successfully!")
        return vectorstore

    if status_callback:
        status_callback("[>>>] No vectorstore found. Loading PDFs...")

    if not os.path.exists(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        raise FileNotFoundError(
            "No PDFs found and no vectorstore found."
        )

    all_documents = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf_file)
        if status_callback:
            status_callback(f"[>>>] Loading: {pdf_file}")
        loader = PyMuPDFLoader(path)
        all_documents.extend(loader.load())

    if status_callback:
        status_callback(f"[:::] Loaded {len(all_documents)} pages. Splitting...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_documents)

    if status_callback:
        status_callback(f"[###] {len(chunks)} chunks. Embedding (one-time)...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    vectorstore.save_local(FAISS_PATH)

    if status_callback:
        status_callback("[=V=] Vectorstore saved to disk!")

    return vectorstore


def build_rag_chain(groq_api_key: str, vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=groq_api_key
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful physics tutor. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information on that."

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