from pathlib import Path

OUTPUT_DIR = "outputs"
DATA_DIR = "data"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_SAVE_EXTRACTED_IMAGES = 100
MAX_IMAGE_SUMMARIES = 75

TOP_K_RESULTS = 10
TOP_K_IMAGES_PER_QUERY = 3
