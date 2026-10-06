import os
import tempfile
import streamlit as st

import ingest
import retrieval  
st.set_page_config(page_title="NERO.AI", layout="centered")
st.title("NERO.AI")
st.caption("Upload a PDF, process it, then ask anything about it.")


if "child_chunks" not in st.session_state:
    st.session_state.child_chunks = None
if "history" not in st.session_state:
    st.session_state.history = []


pdf = st.file_uploader("Upload PDF", type="pdf")

if st.button("▶ PROCESS", disabled=pdf is None):
    with st.spinner("PROCESSING...THIS MIGHT TAKE A WHILE"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getbuffer())
            tmp_path = tmp.name
        try:
            parents, children = ingest.load_chunk(tmp_path)
            ingest.store(parents, children)
            st.session_state.child_chunks = children
        finally:
            os.remove(tmp_path)
    st.success(f"Done: {len(parents)} parents, {len(children)} children")

st.divider()

# ---------- 2. ASK ----------
question = st.text_input("Your question", placeholder="ask anything about the document...")

if st.button("▶ ASK NERO", disabled=not question.strip()):
    if st.session_state.child_chunks is None:
        st.warning("Process a PDF first.")
    else:
        with st.spinner("Thinking..."):
            try:
                # ADAPT: match your real function + return values.
                # Ideal: returns (answer_text, list_of_parent_docs)
                answer, sources = retrieval.generate_answer(
                    question, st.session_state.child_chunks
                )
                st.session_state.history.append((question, answer, sources))
            except Exception as e:
                st.error(f"Something broke: {e}")

# ---------- 3. CHAT HISTORY ----------
for q, a, sources in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**NERO:** {a}")
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i}** (page {page})")
                st.caption(doc.page_content[:400] + "...")
    st.divider()