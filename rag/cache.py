from pathlib import Path
from .config import CACHE_DIR


def get_image_cache_path(pdf_path: str) -> Path:
    pdf_name = Path(pdf_path).stem
    return CACHE_DIR / f"image_cache_{pdf_name}.json"
