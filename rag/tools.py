from pathlib import Path
import os
import shutil
import pytesseract


def setup_tools():
    # Tesseract
    tesseract_exe = Path(__file__).parent.parent / "tools" / "tesseract" / "tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)

    import unstructured_pytesseract.pytesseract as upyt
    upyt.tesseract_cmd = pytesseract.pytesseract.tesseract_cmd

    if not tesseract_exe.exists():
        raise FileNotFoundError(f"Tesseract not found at {tesseract_exe}")

    # Poppler
    poppler_bin = str(Path(__file__).parent.parent / "tools" / "poppler" / "bin")
    os.environ["PATH"] += os.pathsep + poppler_bin

    # Debug prints
    print("[POPPLER] pdfinfo =", shutil.which("pdfinfo"))
    print("[POPPLER] pdftoppm =", shutil.which("pdftoppm"))
    print("[TESSERACT] using:", pytesseract.pytesseract.tesseract_cmd)
