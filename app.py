import os
import shutil
import streamlit as st
from config.config import DATA_DIR, CHROMA_DIR
from models.embeddings import get_embedding_model
from models.llm import get_llm

# Page configuration 
st.set_page_config(
    page_title="KrishiSahayak - Agri Scheme Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

_MODEL_OPTIONS = {
    "gemini": ["gemini-3-flash-preview"],
    "openai": ["gpt-3.5-turbo"],
    "groq":   ["llama-3.1-8b-instant"],
}

@st.cache_resource(show_spinner="Loading and chunking documents...")
def _load_kb_chunks():
    """Load and chunk local knowledge documents once per session."""
    try:
        from utils.document_loader import load_and_chunk

        os.makedirs(DATA_DIR, exist_ok=True)
        return load_and_chunk(DATA_DIR)
    except Exception as e:
        st.warning(f"Document loading failed: {e}")
        return []

@st.cache_resource(show_spinner="Loading knowledge base…")
def _init_vector_store():
    """
    Load the existing ChromaDB index, or build it from documents in data/.
    Result is cached for the entire Streamlit session to avoid repeated I/O.
    """
    try:
        from utils.vector_store import load_vector_store, build_vector_store

        embed_model = get_embedding_model()

        # Prefer loading an already-built index
        vs = load_vector_store(embed_model)
        if vs is not None:
            return vs

        # Build from documents in the data/ directory
        chunks = _load_kb_chunks()
        if not chunks:
            return None

        return build_vector_store(chunks, embed_model)

    except Exception as e:
        st.warning(f"Knowledge base could not be initialised: {e}")
        return None

def _rebuild_vector_store():
    """Delete the on-disk ChromaDB index and clear the Streamlit cache."""
    try:
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        _load_kb_chunks.clear()
        _init_vector_store.clear()
    except Exception as e:
        st.error(f"Could not rebuild knowledge base: {e}")

def _render_assistant_message(content: str):
    """Render assistant output with sources tucked into a dropdown."""
    text = str(content or "")

    if "Sources consulted:" not in text:
        st.markdown(text)
        return

    answer_text, _, sources_block = text.rpartition("Sources consulted:")
    answer_text = answer_text.rstrip()
    if answer_text.endswith("---"):
        answer_text = answer_text[:-3].rstrip()

    if answer_text:
        st.markdown(answer_text)

    cleaned_sources = sources_block.strip()
    if cleaned_sources.startswith("---"):
        cleaned_sources = cleaned_sources[3:].lstrip()
    with st.expander("Sources consulted", expanded=False):
        if cleaned_sources:
            st.markdown(cleaned_sources)
        else:
            st.caption("No sources available.")

def _append_chat_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def _init_session_state():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("response_mode", "Detailed")
    st.session_state.setdefault("use_web", True)

def _chat_page(llm, vectorstore, kb_chunks, response_mode: str, use_web: bool):
    st.title("🌾 KrishiSahayak")
    st.caption("Your AI guide to Indian Government Agricultural Schemes")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(msg["content"])
            else:
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about any government scheme for farmers…"):
        _append_chat_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and web for relevant information…"):
                try:
                    from utils.rag_chain import get_rag_response
                    reply = get_rag_response(
                        query=prompt,
                        chat_history=st.session_state.messages,
                        llm=llm,
                        vectorstore=vectorstore,
                        kb_chunks=kb_chunks,
                        use_web_search=use_web,
                        response_mode=response_mode,
                    )
                except Exception as e:
                    reply = f"An error occurred while generating the response: {e}"
                _render_assistant_message(reply)

            _append_chat_message("assistant", reply)

def main():
    _init_session_state()

    with st.sidebar:
        st.title("🌾 KrishiSahayak")
        st.caption("Agri Scheme Assistant")
        st.divider()

        with st.expander("Response Mode", expanded=False):
            mode_options = ["Concise", "Detailed"]
            mode_help = (
                "**Concise**: short, direct answer (3-5 sentences)\n\n"
                "**Detailed**: fuller explanation with eligibility, steps, and contacts"
            )
            selected_mode = st.segmented_control(
                "Mode",
                mode_options,
                default=st.session_state.response_mode,
                selection_mode="single",
                help=mode_help,
            )
            response_mode = selected_mode or st.session_state.response_mode
            st.session_state.response_mode = response_mode
       
        with st.expander("Live Web Search", expanded=False):
            web_mode = st.segmented_control(
                "Web search",
                ["Off", "On"],
                default="On" if st.session_state.use_web else "Off",
                selection_mode="single",
                help="Supplements knowledge base with real-time web results (DuckDuckGo or Serper).",
            )
            use_web = (web_mode or ("On" if st.session_state.use_web else "Off")) == "On"
            st.session_state.use_web = use_web
        
        with st.expander("Model Settings", expanded=False):
            provider = st.selectbox("Provider", list(_MODEL_OPTIONS.keys()), index=0)
            model = st.selectbox("Model", _MODEL_OPTIONS[provider])    
        
        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Initialise LLM 
    llm = None
    try:
        llm = get_llm(provider=provider, model=model)
    except Exception as e:
        st.error(f"Could not load the selected model: {e}  \nPlease check your API key.")

    # Initialise vector store 
    vectorstore = _init_vector_store()
    kb_chunks = _load_kb_chunks()

    if st.session_state.pop("kb_rebuilt_notice", False):
        if vectorstore is not None:
            try:
                st.toast("Knowledge base rebuilt and loaded successfully.", icon="✅")
            except Exception:
                st.success("Knowledge base rebuilt and loaded successfully.")
        else:
            try:
                st.toast("Rebuild completed, but no documents were found in data/.", icon="⚠️")
            except Exception:
                st.warning("Rebuild completed, but no documents were found in data/.")

    # Chat Interface with RAG
    if llm:
        _chat_page(
            llm,
            vectorstore,
            kb_chunks,
            response_mode,
            use_web,
        )
    else:
        st.info("Configure a valid API key in the sidebar to start chatting.")

if __name__ == "__main__":
    main()
