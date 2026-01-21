# rag_backend.py
from pathlib import Path
import os
import base64
import json
import shutil
import pytesseract
from base64 import b64decode
import uuid

from unstructured.partition.pdf import partition_pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage


# Config
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_SAVE_EXTRACTED_IMAGES = 100
MAX_IMAGE_SUMMARIES = 75

TOP_K_RESULTS = 10
TOP_K_IMAGES_PER_QUERY = 3


# Utils
def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def save_base64_image(base64_code: str, filename: str) -> str:
    image_bytes = base64.b64decode(base64_code)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


# Tooling setup (Tesseract/Poppler)
def setup_tools():
    # Tesseract
    tesseract_exe = Path(__file__).parent / "tools" / "tesseract" / "tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)

    import unstructured_pytesseract.pytesseract as upyt
    upyt.tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

    if not tesseract_exe.exists():
        raise FileNotFoundError(f"Tesseract not found at {tesseract_exe}")

    # Poppler
    poppler_bin = str(Path(__file__).parent / "tools" / "poppler" / "bin")
    os.environ["PATH"] += os.pathsep + poppler_bin

    # Debug prints
    print("[POPPLER] pdfinfo =", shutil.which("pdfinfo"))
    print("[POPPLER] pdftoppm =", shutil.which("pdftoppm"))
    print("[TESSERACT] using:", pytesseract.pytesseract.tesseract_cmd)


# Caching: images + summaries
def get_image_cache_path(pdf_path: str) -> Path:
    # Unique cache per PDF name (avoid collisions)
    pdf_name = Path(pdf_path).stem
    return CACHE_DIR / f"image_cache_{pdf_name}.json"


def get_or_create_image_summaries(chunks, image_cache_path: Path, vision_model: str = "llava:7b"):
    if image_cache_path.exists():
        print("Loading cached image summaries")
        try:
            with open(image_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    print("Extracting and summarizing images (first run only)...")

    images_b64 = get_images_base64(chunks)
    images_b64 = images_b64[:max(MAX_SAVE_EXTRACTED_IMAGES, MAX_IMAGE_SUMMARIES)]

    # Save extracted images (debug)
    if images_b64:
        saved = 0
        for i, img_b64 in enumerate(images_b64[:MAX_SAVE_EXTRACTED_IMAGES]):
            if not img_b64:
                continue
            save_base64_image(img_b64, f"extracted_image_{i}.png")
            saved += 1

        print("-" * 40)
        print(f"Saved {saved} extracted images to {OUTPUT_DIR}")
        print("-" * 40)

    if not images_b64:
        with open(image_cache_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    images_for_summary = images_b64[:MAX_IMAGE_SUMMARIES]

    prompt_template = """
Describe this image for retrieval in a car owner's manual.
Include:
- what component/part it shows (battery, fuse box, jack points, etc.)
- where it is located (engine bay, trunk, under seat, dashboard, etc.)
- any steps/actions shown (remove, loosen, unplug, lift, etc.)
- any warnings/cautions visible
Keep it concise but keyword-rich.
"""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,{image}"},
            ],
        )
    ]
    prompt_img = ChatPromptTemplate.from_messages(messages)
    image_chain = prompt_img | ChatOllama(model=vision_model) | StrOutputParser()

    summaries = image_chain.batch(images_for_summary, {"max_concurrency": 3})

    image_cache = []
    for i, summary in enumerate(summaries):
        image_cache.append({"image_base64": images_for_summary[i], "summary": summary})

    with open(image_cache_path, "w", encoding="utf-8") as f:
        json.dump(image_cache, f, indent=2)

    return image_cache


# Ingest: build retriever
def ingest_manual(
    pdf_path: str,
    use_images: bool = True,
    text_model: str = "llama3.2:1b",
    vision_model: str = "llava:7b",
    progress_cb=None,
):
    """
    Build summaries + vector index + docstore.
    Returns: (retriever, settings_dict)
    """
    def _progress(pct: int, msg: str):
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    _progress(2, "Preparing output folders...")
    ensure_dirs()

    _progress(5, "Setting up tools (Tesseract/Poppler)...")
    setup_tools()

    pdf_path = str(pdf_path)
    image_cache_path = get_image_cache_path(pdf_path)

    # Partition (FAST by default so it doesn't hang)
    _progress(12, "Partitioning PDF (fast)...")
    partition_kwargs = dict(
        filename=pdf_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        strategy="fast",
    )

    chunks = partition_pdf(**partition_kwargs)

    # Separate tables, text
    _progress(25, "Separating text and tables...")
    tables = []
    texts = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    # Summarize text + tables
    _progress(40, "Summarizing text and tables...")
    prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary".
Just give the summary.

Table or text chunk: {element}
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOllama(model=text_model, temperature=0.5)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3}) if texts else []
    tables_html = [t.metadata.text_as_html for t in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3}) if tables_html else []

    # Images (use cache; NOTE: image extraction needs hi_res, so we do a second pass only if needed)
    image_cache = []
    images = []
    image_summaries = []

    if use_images:
        # Load cache if it exists
        if image_cache_path.exists():
            _progress(50, "Loading cached image summaries...")
            try:
                with open(image_cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    image_cache = data
            except Exception:
                image_cache = []

        # If no cache, do a hi_res pass JUST for images/summaries
        if not image_cache:
            _progress(55, "Extracting images (hi_res) + summarizing (first run)...")
            hi_res_kwargs = dict(
                filename=pdf_path,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=True,
            )
            hi_res_chunks = partition_pdf(**hi_res_kwargs)

            image_cache = get_or_create_image_summaries(
                hi_res_chunks,
                image_cache_path=image_cache_path,
                vision_model=vision_model,
            )

        images = [c["image_base64"] for c in image_cache if c.get("image_base64")]
        image_summaries = [c["summary"] for c in image_cache if c.get("summary")]

    # Vectorstore + retriever
    _progress(70, "Building vector index...")
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    retriever.search_kwargs = {"k": TOP_K_RESULTS}

    # Add text
    _progress(80, "Indexing text summaries...")
    if text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
        )
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    _progress(88, "Indexing table summaries...")
    if table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)]
        )
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add images
    if use_images and image_summaries:
        _progress(94, "Indexing image summaries...")
        img_ids = [str(uuid.uuid4()) for _ in image_summaries]
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: img_ids[i]}) for i, s in enumerate(image_summaries)]
        )
        retriever.docstore.mset(list(zip(img_ids, images)))

    settings = {
        "pdf_path": pdf_path,
        "use_images": use_images,
        "text_model": text_model,
        "vision_model": vision_model,
        "image_cache_path": str(image_cache_path),
    }

    _progress(100, "Ingestion complete.")
    return retriever, settings


# Build chain + ask
def build_chain(
    retriever,
    use_images: bool = True,
    answer_model: str = "llama3.2:3b",
):

    # Pull more candidates so we can choose text over images
    try:
        retriever.search_kwargs = {"k": max(TOP_K_RESULTS * 4, 20)}
    except Exception:
        pass

    MAX_TEXT_CHUNKS = 8
    MAX_IMAGE_ATTACH = TOP_K_IMAGES_PER_QUERY

    PROCEDURE_HINTS = (
        "how do", "how to", "steps", "procedure", "remove", "install", "replace",
        "disconnect", "reconnect", "tighten", "loosen", "torque", "battery removal",
        "service plug", "hv battery", "hybrid battery"
    )

    VISUAL_HINTS = (
        "where", "location", "located", "diagram", "picture", "image", "shown",
        "what does it look like", "identify", "which one", "point to"
    )

    def is_probably_base64(s: str) -> bool:
        if not isinstance(s, str):
            return False
        if len(s) < 200:
            return False
        if any(c.isspace() for c in s):
            return False
        try:
            b64decode(s, validate=True)
            return True
        except Exception:
            return False

    def chunk_to_text(obj) -> str:
        if hasattr(obj, "text") and isinstance(getattr(obj, "text"), str):
            return obj.text
        if hasattr(obj, "metadata") and getattr(obj.metadata, "text_as_html", None):
            return str(obj.metadata.text_as_html)
        return str(obj)

    def parse_docs(docs_):
        images_b64 = []
        text_chunks = []

        for d in docs_:
            content = getattr(d, "page_content", d)

            # Images are stored as raw base64 strings in docstore
            if isinstance(content, str) and is_probably_base64(content):
                if len(images_b64) < MAX_IMAGE_ATTACH:
                    images_b64.append(content)
                continue

            text_chunks.append(d)

        # Prefer text
        text_chunks = text_chunks[:MAX_TEXT_CHUNKS]
        return {"images": images_b64, "texts": text_chunks}

    def should_attach_images(question: str) -> bool:
        q = (question or "").lower()

        # If it’s a procedure/steps question, DO NOT attach images by default.
        if any(h in q for h in PROCEDURE_HINTS):
            return False

        # If it’s visual/location/identification, images can help.
        if any(h in q for h in VISUAL_HINTS):
            return True

        # Default: don’t attach (keeps answers grounded in text)
        return False

    def build_messages(kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        for el in docs_by_type.get("texts", []):
            context_text += chunk_to_text(el).strip() + "\n\n"

        rules = (
            "You must answer using the TEXT CONTEXT below.\n"
            "- For step-by-step procedures, rely on text instructions.\n"
            "- Images are optional supporting evidence only.\n"
            "- If the text contains the procedure, give the steps clearly.\n"
            "- Do NOT say you cannot answer just because an image is blurry.\n"
            "- If the text does NOT contain the answer, say you cannot find it.\n"
        )

        prompt = f"""{rules}

TEXT CONTEXT:
{context_text if context_text.strip() else "[No relevant text retrieved]"}

QUESTION:
{user_question}
"""

        prompt_content = [{"type": "text", "text": prompt}]

        vision_capable = ("llava" in answer_model.lower()) or ("vision" in answer_model.lower())
        attach_images = use_images and vision_capable and docs_by_type.get("images") and should_attach_images(user_question)

        if attach_images:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}
                )

        return [HumanMessage(content=prompt_content)]

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(RunnableLambda(build_messages) | ChatOllama(model=answer_model) | StrOutputParser())
    )

    return chain_with_sources



def ask(chain, query: str):
    """
    Returns: (answer_text, images_base64_list)
    """
    result = chain.invoke(query)
    answer = result["response"]
    images_b64 = result["context"]["images"] if "context" in result else []
    return answer, images_b64
