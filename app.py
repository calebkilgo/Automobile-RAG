import streamlit as st
from pathlib import Path
from rag_backend import ingest_manual, build_chain, ask

st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("./assets/logo.png", width=400)

with col2:
    st.title("Automobile Manual RAG")
    st.caption("Ask questions about your car's manual, including images.")

st.divider()

# Session state init
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# Sidebar
with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload PDF manual", type=["pdf"])
    if uploaded_pdf:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        pdf_path = data_dir / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getbuffer())
        st.session_state.pdf_path = str(pdf_path)
        st.success("PDF uploaded")

    if st.button("Ingest manual"):
        if not st.session_state.pdf_path:
            st.error("Upload a PDF first.")
        else:
            with st.spinner("Ingesting manual..."):
                retriever, _settings = ingest_manual(st.session_state.pdf_path, use_images=True)
                st.session_state.chain = build_chain(retriever, use_images=True, answer_model="llava:7b")
                st.session_state.ingested = True
            st.success("Ingestion complete")

    st.divider()

    if st.button("Clear chat"):
        st.session_state.chat = []

# Render chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("images"):
            cols = st.columns(min(3, len(msg["images"])))
            for i, img_b64 in enumerate(msg["images"][:3]):
                cols[i].image(f"data:image/png;base64,{img_b64}", use_container_width=True)


# Chat input
query = st.chat_input("Type your question here")

if query:
    if not st.session_state.ingested or st.session_state.chain is None:
        st.error("Ingest the manual first.")
    else:
        st.session_state.chat.append({"role": "user", "content": query})

        with st.spinner("Generating answer..."):
            answer, images_b64 = ask(st.session_state.chain, query)

        st.session_state.chat.append(
            {"role": "assistant", "content": answer, "images": images_b64}
        )

        st.rerun()
