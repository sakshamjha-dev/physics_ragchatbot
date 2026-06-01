import streamlit as st
from rag import get_embedding_model, load_or_create_vectorstore, build_rag_chain

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Physics RAG Tutor",
    page_icon="[*.*]",
    layout="centered"
)

st.title("Physics RAG Tutor")
st.caption("Powered by HuggingFace Embeddings · Groq LLaMA3 · FAISS")

# ── Sidebar: API key input ────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",           # hides the key as user types
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )
    st.markdown("[Get a free Groq API key →](https://console.groq.com)")
    st.divider()
    st.markdown("**About**")
    st.markdown("Ask anything from University-physics-volume 1 , 2 & 3.")

# ── Session state: cache vectorstore across reruns ────────────
@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """
    Cached once per session — embedding model loads once,
    vectorstore loads from disk or builds once.
    """
    status_box = st.empty()
    logs = []

    def update_status(msg):
        logs.append(msg)
        status_box.info("\n\n".join(logs))

    embedding_model = get_embedding_model()
    vs = load_or_create_vectorstore(embedding_model, status_callback=update_status)
    status_box.empty()  # clear status messages after loading
    return vs

# ── Load vectorstore on startup ───────────────────────────────
with st.spinner("Initializing knowledge base..."):
    try:
        vectorstore = get_vectorstore()
        st.success("[=V=] Knowledge base ready!")
    except FileNotFoundError as e:
        st.error(f"[=X=] {e}\n\nPlease add PDF files to the `pdfs/` folder.")
        st.stop()

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
if question := st.chat_input("Ask a physics question..."):

    if not groq_api_key:
        st.warning("[!/!] Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and stream answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = build_rag_chain(groq_api_key, vectorstore)
                response = chain.invoke(question)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"[=X=] Error: {e}"
                st.error(err)
