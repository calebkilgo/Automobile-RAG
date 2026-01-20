from pathlib import Path
import os
import base64
import json
import shutil
import pytesseract
from base64 import b64decode

from unstructured.partition.pdf import partition_pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

import uuid
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
    return CACHE_DIR / "image_cache.json"


def get_or_create_image_summaries(chunks, image_cache_path: Path, vision_model: str = "llava:7b"):
    if image_cache_path.exists():
        print("Loading cached image summaries")
        with open(image_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
):
    """
    Build summaries + vector index + docstore.
    Returns: (retriever, settings_dict)
    """
    ensure_dirs()
    setup_tools()

    pdf_path = str(pdf_path)
    image_cache_path = get_image_cache_path(pdf_path)

    # Partition
    partition_kwargs = dict(
        filename=pdf_path,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    if use_images and not image_cache_path.exists():
        partition_kwargs.update(
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
        )
    else:
        partition_kwargs.update(strategy="fast")

    chunks = partition_pdf(**partition_kwargs)

    # Separate tables, text
    tables = []
    texts = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)

    # Summarize text + tables
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

    # Images (cached)
    image_cache = []
    images = []
    image_summaries = []
    if use_images:
        image_cache = get_or_create_image_summaries(chunks, image_cache_path, vision_model=vision_model)
        images = [item["image_base64"] for item in image_cache]
        image_summaries = [item["summary"] for item in image_cache]

    # Vectorstore + retriever
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)
    retriever.search_kwargs = {"k": TOP_K_RESULTS}

    # Add text
    if text_summaries:
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(text_summaries)]
        )
        retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    if table_summaries:
        table_ids = [str(uuid.uuid4()) for _ in tables]
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: table_ids[i]}) for i, s in enumerate(table_summaries)]
        )
        retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add images
    if use_images and image_summaries:
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

    return retriever, settings


# Build chain + ask
def build_chain(retriever, use_images: bool = True, answer_model: str = "llama3.2:3b"):  # ✅ UPDATED
    def parse_docs(docs_):
        b64_list = []
        text_list = []
        for doc in docs_:
            content = getattr(doc, "page_content", doc)

            if isinstance(content, str):
                try:
                    b64decode(content, validate=True)
                    if len(b64_list) < TOP_K_IMAGES_PER_QUERY:
                        b64_list.append(content)
                    continue
                except Exception:
                    pass

            text_list.append(doc)

        return {"images": b64_list, "texts": text_list}

    def build_prompt(kwargs):
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if docs_by_type["texts"]:
            for text_element in docs_by_type["texts"]:
                context_text += getattr(text_element, "text", str(text_element)) + "\n"

        prompt_template = f"""
            Answer the question based only on the following context, which can include text, tables, and images.
            Context: {context_text}
            Question: {user_question}
            """

        prompt_content = [{"type": "text", "text": prompt_template}]

        vision_capable = ("llava" in answer_model.lower()) or ("vision" in answer_model.lower())

        if use_images and vision_capable and docs_by_type["images"]:
            for image in docs_by_type["images"]:
                prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"})

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(RunnableLambda(build_prompt) | ChatOllama(model=answer_model) | StrOutputParser())
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
