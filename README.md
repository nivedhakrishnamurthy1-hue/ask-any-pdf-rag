# Ask Any PDF

Upload a PDF, ask questions, and get short answers with **page citations**. It answers only from your document and says so when it can't find the answer.

**Live demo:** https://ask-any-pdf.streamlit.app/

## How it works

PDF → text per page → overlapping chunks → MiniLM embeddings → FAISS search → top passages → GPT-OSS 120B (Groq) → validated JSON answer with page numbers

- If no passage is similar enough to the question, the app refuses without calling the LLM.
- The model must reply in JSON, and any page citation it was not given is discarded.