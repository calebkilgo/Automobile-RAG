import json
import uuid
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

from .config import TOP_K_RESULTS
from .tools import setup_tools
from .cache import get_image_cache_path
from .image_utils import get_or_create_image_summaries


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
    tables, texts = [], []
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

    # Images (cache; hi_res pass only if needed)
    image_cache = []
    images, image_summaries = [], []

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
