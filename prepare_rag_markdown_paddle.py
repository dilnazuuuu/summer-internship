# Extends prepare_rag_markdown.py with EasyOCR support for scanned PDFs and images.
# Office-style files still use the base script; image/PDF inputs use EasyOCR text recognition.
# The old "paddle" names are kept in public APIs so app.py and old CLI commands keep working.

import argparse
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

from pdf2image import convert_from_path

from prepare_rag_markdown import (
    EXTENSIONS as OFFICE_EXTENSIONS,
    clean_markdown,
    convert_to_raw_markdown,
    tqdm,
    validate_args as validate_office_args,
)


os.environ.setdefault("OMP_THREAD_LIMIT", "1")

PADDLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXTENSIONS = OFFICE_EXTENSIONS | PADDLE_EXTENSIONS
PDF_OCR_DPI = int(os.environ.get("OCR_PDF_DPI", "150"))
PADDLE_OCR_TIMEOUT_S = int(os.environ.get("PADDLE_OCR_TIMEOUT_S", "120"))
EASYOCR_MODEL_DIR = Path(os.environ.get("EASYOCR_MODEL_DIR", "/tmp/easyocr"))

LANGUAGE_MAP = {
    "ru": ["ru", "en"],
    # EasyOCR 1.7.2 does not ship a Kazakh model, so use Cyrillic-compatible
    # Russian recognition plus English as the practical fallback.
    "kk": ["ru", "en"],
    "en": ["en"],
}

_EASYOCR_READERS = {}


def reader_key(args) -> tuple[str, str]:
    return (getattr(args, "lang", "ru"), getattr(args, "device", "cpu"))


def create_easyocr_reader(easyocr, languages: list[str], use_gpu: bool):
    EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    user_network_dir = EASYOCR_MODEL_DIR / "user_network"
    user_network_dir.mkdir(parents=True, exist_ok=True)
    return easyocr.Reader(
        languages,
        gpu=use_gpu,
        model_storage_directory=str(EASYOCR_MODEL_DIR),
        user_network_directory=str(user_network_dir),
        verbose=False,
    )


def get_text_ocr(args):
    key = reader_key(args)
    if key in _EASYOCR_READERS:
        return _EASYOCR_READERS[key]

    try:
        import easyocr
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EasyOCR is not installed. Install it first: python -m pip install easyocr"
        ) from exc

    lang_code = getattr(args, "lang", "ru")
    languages = LANGUAGE_MAP.get(lang_code, ["ru", "en"])
    use_gpu = getattr(args, "device", "cpu") == "gpu"

    reader = create_easyocr_reader(easyocr, languages, use_gpu)

    _EASYOCR_READERS[key] = reader
    return reader


def configure_paddle_pipeline(args):
    # Compatibility no-op. EasyOCR initialises lazily in get_text_ocr().
    return None


def extract_text_from_easyocr_result(result) -> str:
    lines = []
    for item in result or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[1]).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def run_ocr_with_timeout(reader, image, label: str):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(reader.readtext, image)
    try:
        return future.result(timeout=PADDLE_OCR_TIMEOUT_S) or []
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"OCR timeout after {PADDLE_OCR_TIMEOUT_S}s on {label}") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def ocr_image_to_text(reader, image, label: str) -> str:
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        image = np.array(image, dtype=np.uint8)
    image = np.ascontiguousarray(image)
    output = run_ocr_with_timeout(reader, image, label)
    return extract_text_from_easyocr_result(output)


def pdf_page_count(src: Path) -> int:
    try:
        import fitz

        with fitz.open(str(src)) as doc:
            return len(doc)
    except Exception:
        return 0


def paddle_pdf_ocr_to_raw_markdown(src: Path, args) -> str:
    reader = get_text_ocr(args)
    page_count = pdf_page_count(src)
    if page_count <= 0:
        raise ValueError("PDF has no pages or could not be opened")

    print(f"OCR started: {src.name} ({page_count} pages)", flush=True)
    parts = [f"# {src.stem}"]
    for page_num in range(1, page_count + 1):
        print(f"OCR page {page_num}/{page_count}: {src.name}", flush=True)
        images = convert_from_path(
            str(src),
            dpi=PDF_OCR_DPI,
            first_page=page_num,
            last_page=page_num,
            fmt="png",
            thread_count=1,
            timeout=PADDLE_OCR_TIMEOUT_S,
        )
        if not images:
            continue
        text = ocr_image_to_text(reader, images[0], f"{src.name} page {page_num}")
        if text:
            parts.append(f"## Страница {page_num}\n\n{text}")
        del images
    print(f"OCR finished: {src.name}", flush=True)
    return "\n\n".join(parts)


def paddle_image_ocr_to_raw_markdown(src: Path, args) -> str:
    reader = get_text_ocr(args)
    print(f"OCR started: {src.name}", flush=True)
    text = ocr_image_to_text(reader, str(src), src.name)
    print(f"OCR finished: {src.name}", flush=True)
    if not text:
        return f"# {src.stem}"
    return f"# {src.stem}\n\n{text}"


def paddle_to_raw_markdown(src: Path, args) -> str:
    if src.suffix.lower() == ".pdf":
        return paddle_pdf_ocr_to_raw_markdown(src, args)
    return paddle_image_ocr_to_raw_markdown(src, args)


def convert_file_to_raw_markdown(src: Path, args, office_config: dict) -> tuple[str, str]:
    ext = src.suffix.lower()
    if ext in PADDLE_EXTENSIONS:
        return paddle_to_raw_markdown(src, args), "easyocr"
    raw, mode = convert_to_raw_markdown(src, office_config)
    return raw, mode


def process_file(src: Path, input_dir: Path, output_dir: Path, args, office_config: dict):
    empty_stats = {
        "footers_removed": 0,
        "paras_dropped": 0,
        "tables_converted": 0,
        "picture_blocks_dropped": 0,
        "graph_blocks_dropped": 0,
        "figure_captions_dropped": 0,
    }
    try:
        rel = src.relative_to(input_dir)
        out_dir = output_dir / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{src.stem}.md"

        if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
            return str(src), "skip", "exists", empty_stats

        raw_markdown, mode = convert_file_to_raw_markdown(src, args, office_config)
        cleaned, stats = clean_markdown(raw_markdown, src.stem)
        if not cleaned.strip():
            raise ValueError("Resulting markdown is empty after EasyOCR/cleaning")

        out_file.write_text(cleaned, encoding="utf-8")
        return str(src), "ok", mode, stats
    except Exception as exc:
        return str(src), "fail", f"{type(exc).__name__}: {exc}", empty_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert documents to clean RAG Markdown using EasyOCR."
    )
    parser.add_argument("--input", "-i", required=True, help="Input folder with source files.")
    parser.add_argument("--output", "-o", required=True, help="Output folder for .md files.")
    parser.add_argument("--failed-log", help="Path for failed conversions log.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .md files.")
    parser.add_argument("--libreoffice", help="Path to LibreOffice soffice executable for .doc/.rtf.")
    parser.add_argument(
        "--lang",
        default="ru",
        choices=sorted(LANGUAGE_MAP),
        help="OCR language. Default: ru. Options: ru, kk, en.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Inference device. Default: cpu.",
    )

    # Kept for compatibility with existing app.py and older shell commands.
    parser.add_argument("--paddle-mode", default="ocr", help=argparse.SUPPRESS)
    parser.add_argument("--use-doc-orientation-classify", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-doc-unwarping", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use-textline-orientation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-table-recognition", dest="use_table_recognition", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--use-seal-recognition", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-formula-recognition", dest="use_formula_recognition", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(use_table_recognition=True, use_formula_recognition=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    failed_log = Path(args.failed_log).expanduser().resolve() if args.failed_log else output_dir / "failed_easyocr_convert.txt"

    if not input_dir.exists():
        print(f"ERROR: input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = [
        file
        for file in input_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in EXTENSIONS
    ]

    office_args = argparse.Namespace(
        overwrite=args.overwrite,
        no_ocr=True,
        poppler_path=None,
        tesseract_cmd=None,
        libreoffice=args.libreoffice,
    )
    office_config = validate_office_args(
        office_args,
        [file for file in files if file.suffix.lower() in OFFICE_EXTENSIONS - {".pdf"}],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Files: {len(files)} | OCR: EasyOCR | lang: {args.lang} | device: {args.device} | workers: 1")
    if not files:
        print(f"Output: {output_dir}")
        print("No supported files found.")
        sys.exit(0)

    if any(file.suffix.lower() in PADDLE_EXTENSIONS for file in files):
        print("Loading EasyOCR model...", flush=True)
        get_text_ocr(args)
        print("EasyOCR model ready.", flush=True)

    ok = skip = fail = 0
    totals = {
        "footers_removed": 0,
        "paras_dropped": 0,
        "tables_converted": 0,
        "picture_blocks_dropped": 0,
        "graph_blocks_dropped": 0,
        "figure_captions_dropped": 0,
    }
    failed = []

    with tqdm(total=len(files), unit="file") as progress:
        for file in files:
            src, status, message, stats = process_file(file, input_dir, output_dir, args, office_config)
            if status == "ok":
                ok += 1
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0) + value
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failed.append(f"{src}\t{message}")
            progress.set_postfix(ok=ok, skip=skip, fail=fail)
            progress.update(1)

    print(f"\nOK: {ok} | skipped: {skip} | failed: {fail}")
    print(f"Output: {output_dir}")
    print(f"Tables converted to RAG text: {totals['tables_converted']}")
    print(f"Footers removed: {totals['footers_removed']}")
    print(f"Low-value paragraphs dropped: {totals['paras_dropped']}")
    print(f"Picture OCR blocks dropped: {totals['picture_blocks_dropped']}")
    print(f"Graph OCR blocks dropped: {totals['graph_blocks_dropped']}")
    print(f"Figure captions dropped: {totals['figure_captions_dropped']}")

    if failed:
        failed_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed log: {failed_log}")

    sys.exit(0 if fail == 0 else 2)


if __name__ == "__main__":
    main()
