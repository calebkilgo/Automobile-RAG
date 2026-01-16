import os
from pathlib import Path

from rag_backend import ingest_manual, build_chain, ask

USE_IMAGES = True

def newest_pdf_in_data(data_dir="data"):
    pdfs = [p for p in Path(data_dir).glob("*.pdf")]
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in '{data_dir}'.")
    return str(max(pdfs, key=lambda p: p.stat().st_mtime))

def main():
    pdf_path = newest_pdf_in_data("data")
    print("[PDF] using:", pdf_path)

    retriever, settings = ingest_manual(
        pdf_path=pdf_path,
        use_images=USE_IMAGES,
        text_model="llama3.2",
        vision_model="llava:7b",
    )

    chain = build_chain(
        retriever=retriever,
        use_images=USE_IMAGES,
        answer_model="llava:7b" if USE_IMAGES else "llama3.2",
    )

    print("\nReady. Type questions (exit/quit to stop).")
    while True:
        q = input(">> ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        answer, images_b64 = ask(chain, q)
        print("\nAnswer:\n", answer)

        # Save images to outputs/ for terminal viewing
        if USE_IMAGES and images_b64:
            os.makedirs("outputs", exist_ok=True)
            from rag_backend import save_base64_image
            for i, img in enumerate(images_b64[:3]):
                save_base64_image(img, f"retrieved_image_{i}.png")
            print(f"[INFO] Saved {min(3, len(images_b64))} retrieved images to outputs/")

if __name__ == "__main__":
    main()
