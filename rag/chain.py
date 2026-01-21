from base64 import b64decode

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from .config import TOP_K_RESULTS, TOP_K_IMAGES_PER_QUERY
from .image_utils import is_probably_base64


def build_chain(retriever, use_images: bool = True, answer_model: str = "llama3.2:3b"):
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

            if isinstance(content, str) and is_probably_base64(content):
                if len(images_b64) < MAX_IMAGE_ATTACH:
                    images_b64.append(content)
                continue

            text_chunks.append(d)

        text_chunks = text_chunks[:MAX_TEXT_CHUNKS]
        return {"images": images_b64, "texts": text_chunks}

    def should_attach_images(question: str) -> bool:
        q = (question or "").lower()
        if any(h in q for h in PROCEDURE_HINTS):
            return False
        if any(h in q for h in VISUAL_HINTS):
            return True
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
                prompt_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"})

        return [HumanMessage(content=prompt_content)]

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(RunnableLambda(build_messages) | ChatOllama(model=answer_model) | StrOutputParser())
    )

    return chain_with_sources


def ask(chain, query: str):
    result = chain.invoke(query)
    answer = result["response"]
    images_b64 = result["context"]["images"] if "context" in result else []
    return answer, images_b64
