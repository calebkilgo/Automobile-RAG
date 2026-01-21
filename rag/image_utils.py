import os
import base64
import json
from base64 import b64decode
from typing import List, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from .config import OUTPUT_DIR, MAX_SAVE_EXTRACTED_IMAGES, MAX_IMAGE_SUMMARIES


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_base64_image(base64_code: str, filename: str) -> str:
    image_bytes = base64.b64decode(base64_code)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def get_images_base64(chunks) -> List[str]:
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)) and getattr(el.metadata, "image_base64", None):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def get_or_create_image_summaries(
    chunks,
    image_cache_path,
    vision_model: str = "llava:7b",
):
    """
    Returns list of dicts: [{"image_base64": "...", "summary": "..."}, ...]
    """
    ensure_dirs()

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
