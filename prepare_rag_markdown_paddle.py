# Extends prepare_rag_markdown.py with Tesseract support for scanned PDFs and images.
# Office-style files still use the base script; image/PDF inputs use Tesseract OCR.
# The old "paddle" names are kept in public APIs so app.py and old CLI commands keep working.

import argparse
import gc
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps
import pytesseract
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
PDF_OCR_DPI = int(os.environ.get("OCR_PDF_DPI", "200"))
PADDLE_OCR_TIMEOUT_S = int(os.environ.get("PADDLE_OCR_TIMEOUT_S", "120"))
TESSERACT_CONFIG = os.environ.get("TESSERACT_CONFIG", "--oem 1 --psm 4")

LANGUAGE_MAP = {
    "ru": "rus+eng",
    "kk": "kaz+rus+eng",
    "en": "eng",
}


def configure_paddle_pipeline(args):
    # Compatibility no-op. Tesseract does not need model initialisation.
    return None


def tesseract_lang(args) -> str:
    return LANGUAGE_MAP.get(getattr(args, "lang", "ru"), "rus+eng")


def prepare_tesseract_image(image):
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
    image = ImageOps.grayscale(image)
    return ImageOps.autocontrast(image)


def ocr_image_to_text(image, label: str, args) -> str:
    try:
        image = prepare_tesseract_image(image)
        text = pytesseract.image_to_string(
            image,
            lang=tesseract_lang(args),
            config=TESSERACT_CONFIG,
            timeout=PADDLE_OCR_TIMEOUT_S,
        )
        return text.strip()
    except Exception as exc:
        raise RuntimeError(f"OCR failed on {label}: {exc}") from exc


def pdf_page_count(src: Path) -> int:
    try:
        import fitz

        with fitz.open(str(src)) as doc:
            return len(doc)
    except Exception:
        return 0


def paddle_pdf_ocr_to_raw_markdown(src: Path, args, progress_callback=None) -> str:
    page_count = pdf_page_count(src)
    if page_count <= 0:
        raise ValueError("PDF has no pages or could not be opened")

    print(f"OCR started: {src.name} ({page_count} pages)", flush=True)
    if progress_callback:
        progress_callback(0, page_count)
    parts = [f"# {src.stem}"]
    for page_num in range(1, page_count + 1):
        print(f"OCR page {page_num}/{page_count}: {src.name}", flush=True)
        images = []
        image = None
        try:
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
            image = images[0]
            text = ocr_image_to_text(image, f"{src.name} page {page_num}", args)
            if text:
                parts.append(f"## Страница {page_num}\n\n{text}")
            if progress_callback:
                progress_callback(page_num, page_count)
        finally:
            del image
            del images
            gc.collect()
    print(f"OCR finished: {src.name}", flush=True)
    return "\n\n".join(parts)


def paddle_image_ocr_to_raw_markdown(src: Path, args, progress_callback=None) -> str:
    if progress_callback:
        progress_callback(0, 1)
    print(f"OCR started: {src.name}", flush=True)
    text = ocr_image_to_text(str(src), src.name, args)
    print(f"OCR finished: {src.name}", flush=True)
    if progress_callback:
        progress_callback(1, 1)
    if not text:
        return f"# {src.stem}"
    return f"# {src.stem}\n\n{text}"


def paddle_to_raw_markdown(src: Path, args, progress_callback=None) -> str:
    if src.suffix.lower() == ".pdf":
        return paddle_pdf_ocr_to_raw_markdown(src, args, progress_callback)
    return paddle_image_ocr_to_raw_markdown(src, args, progress_callback)


def convert_file_to_raw_markdown(src: Path, args, office_config: dict, progress_callback=None) -> tuple[str, str]:
    ext = src.suffix.lower()
    if ext in PADDLE_EXTENSIONS:
        return paddle_to_raw_markdown(src, args, progress_callback), "tesseract"
    raw, mode = convert_to_raw_markdown(src, office_config)
    return raw, mode


def process_file(src: Path, input_dir: Path, output_dir: Path, args, office_config: dict, progress_callback=None):
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

        raw_markdown, mode = convert_file_to_raw_markdown(src, args, office_config, progress_callback)
        cleaned, stats = clean_markdown(raw_markdown, src.stem)
        if not cleaned.strip():
            raise ValueError("Resulting markdown is empty after Tesseract/cleaning")

        out_file.write_text(cleaned, encoding="utf-8")
        return str(src), "ok", mode, stats
    except Exception as exc:
        return str(src), "fail", f"{type(exc).__name__}: {exc}", empty_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert documents to clean RAG Markdown using Tesseract."
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
    failed_log = Path(args.failed_log).expanduser().resolve() if args.failed_log else output_dir / "failed_tesseract_convert.txt"

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
    print(f"Files: {len(files)} | OCR: Tesseract | lang: {args.lang} | workers: 1")
    if not files:
        print(f"Output: {output_dir}")
        print("No supported files found.")
        sys.exit(0)

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
