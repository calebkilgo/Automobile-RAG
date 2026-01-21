# app.py
import streamlit as st
from pathlib import Path
from rag_backend import ingest_manual, build_chain, ask

st.set_page_config(layout="wide")

col1, col2 = st.columns([1, 3])

with col1:
    st.image("./assets/logo.png", width=400)

with col2:
    st.title("Automobile Manual RAG")
    st.caption("Ask questions about your car's manual, including images or tables.")

st.divider()

# Session state init
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role": "user"/"assistant", "content": str, "images": [b64]}
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "use_images" not in st.session_state:
    st.session_state.use_images = True

# Sidebar
with st.sidebar:
    uploaded_pdf = st.file_uploader("Upload PDF Manual", type=["pdf"])
    if uploaded_pdf:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        pdf_path = data_dir / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getbuffer())
        st.session_state.pdf_path = str(pdf_path)
        st.success("PDF Upload Complete")

    st.session_state.use_images = st.checkbox(
        "Use images (slower)",
        value=st.session_state.use_images,
    )

    if st.button("Ingest Manual"):
        if not st.session_state.pdf_path:
            st.error("Upload a PDF first.")
        else:
            progress = st.progress(0)
            status = st.empty()

            def cb(pct, msg):
                progress.progress(int(pct))
                status.write(msg)

            retriever, _settings = ingest_manual(
                st.session_state.pdf_path,
                use_images=st.session_state.use_images,
                progress_cb=cb,
            )

            answer_model = "llava:7b" if st.session_state.use_images else "llama3.2:3b"
            st.session_state.chain = build_chain(
                retriever,
                use_images=st.session_state.use_images,
                answer_model=answer_model,
            )

            st.session_state.ingested = True
            st.success("Ingestion Complete")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat = []

# CHAT HISTORY (shows user then assistant)
for msg in st.session_state.chat:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        # Show images only for assistant messages (and only if enabled)
        if (
            st.session_state.use_images
            and role == "assistant"
            and msg.get("images")
        ):
            imgs = msg["images"][:3]
            cols = st.columns(min(3, len(imgs)))
            for i, img_b64 in enumerate(imgs):
                cols[i].image(
                    f"data:image/png;base64,{img_b64}",
                    use_container_width=True
                )

# New Message
query = st.chat_input("Type your question here...")

if query:
    if not st.session_state.ingested or st.session_state.chain is None:
        st.error("Ingest the manual first.")
    else:
        st.session_state.chat.append({"role": "user", "content": query})

        with st.spinner("Generating answer..."):
            answer, images_b64 = ask(st.session_state.chain, query)

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": answer,
                "images": images_b64 if st.session_state.use_images else [],
            }
        )

        st.rerun()
