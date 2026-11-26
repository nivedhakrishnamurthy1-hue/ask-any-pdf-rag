"""Streamlit web app: upload any text-based PDF and ask questions about it.

    streamlit run app.py
"""
import hashlib
import io
import os

import streamlit as st

try:  # Streamlit Community Cloud secrets -> environment variables
    for _name in ("GROQ_API_KEY", "LLM_MODEL"):
        if _name in st.secrets:
            os.environ.setdefault(_name, str(st.secrets[_name]))
except Exception:
    pass

from ragqa import config
from ragqa.answer import ask
from ragqa.pipeline import build_retriever

st.set_page_config(page_title="Ask any PDF", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Reading and indexing the PDF...", max_entries=3, ttl=3600)
def index_pdf(data: bytes, name: str):
    return build_retriever(io.BytesIO(data), name, max_pages=config.MAX_PAGES)


def render(result):
    if result["found"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])
    cols = st.columns(3)
    cols[0].metric("Pages cited", ", ".join(map(str, result["pages"])) or "none")
    top = result["sources"][0]["score"] if result["sources"] else 0.0
    cols[1].metric("Best match score", f"{top:.2f}")
    cols[2].metric("LLM called", "no (gated)" if result.get("gated") else "yes")
    with st.expander("Retrieved passages"):
        for hit in result["sources"]:
            st.markdown(f"**Page {hit['page']}** · similarity {hit['score']:.2f}")
            st.caption(hit["text"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Upload your PDF")
    uploaded = st.file_uploader("Text-based PDF, up to 25 MB", type=["pdf"])
    st.caption(f"Up to {config.MAX_PAGES} pages. Scanned PDFs (photos of pages) are not supported.")

    st.header("2 · Retrieval settings")
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, config.TOP_K)
    min_score = st.slider("Minimum similarity", 0.0, 0.6, config.MIN_SCORE, 0.01,
                          help="Below this the app refuses without calling the LLM.")
    st.divider()
    st.caption(f"LLM: {config.LLM_MODEL} (Groq) · Embeddings: {config.EMBED_MODEL.split('/')[-1]}")
    st.caption("🔒 Your PDF is processed in memory and is not stored. Text excerpts relevant to "
               "your question are sent to the Groq API, so don't upload confidential documents.")

# ---------------------------------------------------------------- document
st.title("📄 Ask any PDF")

if uploaded is None:
    st.write("Upload a PDF in the sidebar and ask questions about it. Answers come **only** "
             "from your document, include **page citations**, and the app says so when it "
             "cannot find the answer instead of guessing.")
    st.stop()

try:
    data = uploaded.getvalue()
    retriever = index_pdf(data, uploaded.name)
    doc_key = hashlib.md5(data).hexdigest()
except Exception as error:
    st.error(f"Could not read this PDF: {error}")
    st.stop()

if st.session_state.get("doc_key") != doc_key:  # new document -> fresh conversation
    st.session_state.doc_key = doc_key
    st.session_state.log = []
    st.session_state.asked = 0

meta = retriever.meta
st.caption(f"Answering from **{meta.get('source')}** · {meta.get('pages')} pages · "
           f"{meta.get('chunks')} chunks. Each question is answered independently.")

limit_reached = st.session_state.asked >= config.MAX_QUESTIONS_PER_SESSION
if limit_reached:
    st.info(f"You've reached the demo limit of {config.MAX_QUESTIONS_PER_SESSION} questions "
            "for this session. Reload the page to start a new session.")

question = st.chat_input("Ask a question about the document", disabled=limit_reached)
if question:
    st.session_state.asked += 1
    with st.spinner("Searching and answering..."):
        try:
            result = ask(question, retriever, k=top_k, min_score=min_score)
        except Exception as error:
            result = {"answer": f"Error: {error}", "found": False, "pages": [],
                      "sources": [], "gated": False}
    st.session_state.log.append({"q": question, "result": result})

for entry in st.session_state.log:
    with st.chat_message("user"):
        st.write(entry["q"])
    with st.chat_message("assistant"):
        render(entry["result"])