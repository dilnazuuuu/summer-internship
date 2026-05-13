# Extends prepare_rag_markdown.py with PaddleOCR support for scanned PDFs and images.
# Office-style files still use the base script; image/PDF inputs can use either
# PP-StructureV3 layout/table extraction or plain OCR text recognition.
# Run with --input and --output; choose --paddle-mode structure or --paddle-mode ocr.

import argparse
import html
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

from prepare_rag_markdown import (
    EXTENSIONS as OFFICE_EXTENSIONS,
    clean_markdown,
    context_prefix,
    convert_to_raw_markdown,
    rows_to_rag_text,
    tqdm,
    validate_args as validate_office_args,
)


os.environ.setdefault("OMP_THREAD_LIMIT", "1")

PADDLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EXTENSIONS = OFFICE_EXTENSIONS | PADDLE_EXTENSIONS
HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ESCAPED_NEWLINE_RE = re.compile(r"\\n")

_PADDLE_PIPELINE = None
_PADDLE_TEXT_OCR = None


# Parses Paddle's HTML tables so they can be converted into the same row-level RAG text.
class TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.current_row = None
        self.current_cell = None
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"} and self.current_row is not None:
            self.current_cell = []
            self.in_cell = True

    def handle_data(self, data):
        if self.in_cell and self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"} and self.in_cell and self.current_cell is not None:
            cell = " ".join("".join(self.current_cell).split())
            self.current_row.append(html.unescape(cell))
            self.current_cell = None
            self.in_cell = False
        elif tag == "tr" and self.current_row is not None:
            if any(cell.strip() for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None


def html_table_to_rag_text(table_html: str, doc_name: str) -> str:
    parser = TableHTMLParser()
    parser.feed(table_html)
    rows = parser.rows
    if len(rows) < 2:
        return ""
    prefix = context_prefix(doc_name, caption="HTML-таблица из PaddleOCR")
    return rows_to_rag_text(rows[0], rows[1:], prefix)


def convert_html_tables(text: str, doc_name: str) -> str:
    def repl(match):
        converted = html_table_to_rag_text(match.group(0), doc_name)
        return f"\n\n{converted}\n\n" if converted else "\n\n"

    return HTML_TABLE_RE.sub(repl, text)


def strip_html_noise(text: str) -> str:
    text = ESCAPED_NEWLINE_RE.sub("\n", text)
    text = convert_html_tables(text, "")
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Paddle models are created lazily so the script only loads the heavy OCR pipeline when needed.
def get_paddle_pipeline():
    global _PADDLE_PIPELINE
    if _PADDLE_PIPELINE is None:
        try:
            from paddleocr import PPStructureV3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it first: "
                "python -m pip install paddlepaddle paddleocr"
            ) from exc
        raise RuntimeError("Paddle pipeline was requested before it was configured")
    return _PADDLE_PIPELINE


def configure_paddle_pipeline(args):
    global _PADDLE_PIPELINE
    if _PADDLE_PIPELINE is not None:
        return
    try:
        from paddleocr import PPStructureV3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install it first: "
            "python -m pip install paddlepaddle paddleocr"
        ) from exc

    _PADDLE_PIPELINE = PPStructureV3(
        lang=args.lang,
        device=args.device,
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
        use_seal_recognition=args.use_seal_recognition,
        use_table_recognition=args.use_table_recognition,
        use_formula_recognition=args.use_formula_recognition,
    )


def get_text_ocr(args):
    global _PADDLE_TEXT_OCR
    if _PADDLE_TEXT_OCR is None:
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install it first: "
                "python -m pip install paddlepaddle paddleocr"
            ) from exc

        _PADDLE_TEXT_OCR = PaddleOCR(
            lang=args.lang,
            device=args.device,
            use_doc_orientation_classify=args.use_doc_orientation_classify,
            use_doc_unwarping=args.use_doc_unwarping,
            use_textline_orientation=args.use_textline_orientation,
        )
    return _PADDLE_TEXT_OCR


# Converts Paddle's structured output into plain Markdown, including table/layout content.
def normalize_concatenated_markdown(markdown_result) -> str:
    if isinstance(markdown_result, tuple):
        markdown_result = markdown_result[0]
    if isinstance(markdown_result, list):
        return "\n\n".join(str(item) for item in markdown_result if str(item).strip())
    return str(markdown_result or "")


def paddle_to_raw_markdown(src: Path, args) -> str:
    if args.paddle_mode == "ocr":
        return paddle_text_ocr_to_raw_markdown(src, args)

    pipeline = get_paddle_pipeline()
    output = pipeline.predict(
        input=str(src),
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_textline_orientation=args.use_textline_orientation,
        use_seal_recognition=args.use_seal_recognition,
        use_table_recognition=args.use_table_recognition,
        use_formula_recognition=args.use_formula_recognition,
    )

    markdown_list = []
    for result in output:
        md_info = getattr(result, "markdown", None)
        if md_info:
            markdown_list.append(md_info)
            continue

        json_info = getattr(result, "json", None)
        if isinstance(json_info, dict):
            parts = []
            for block in json_info.get("parsing_res_list", []):
                content = block.get("block_content")
                if content:
                    parts.append(str(content))
            if parts:
                markdown_list.append({"markdown_text": "\n\n".join(parts)})

    if not markdown_list:
        return ""

    if hasattr(pipeline, "concatenate_markdown_pages"):
        markdown_text = normalize_concatenated_markdown(
            pipeline.concatenate_markdown_pages(markdown_list)
        )
    else:
        markdown_text = "\n\n".join(
            str(item.get("markdown_text", item)) if isinstance(item, dict) else str(item)
            for item in markdown_list
        )

    return strip_html_noise(convert_html_tables(markdown_text, src.stem))


def extract_text_from_paddle_result(result) -> str:
    json_info = getattr(result, "json", None)
    if isinstance(json_info, dict):
        res = json_info.get("res", {})
        rec_texts = res.get("rec_texts")
        if rec_texts:
            return "\n".join(str(text) for text in rec_texts if str(text).strip())

        ocr_res = res.get("ocr_res")
        if isinstance(ocr_res, dict) and ocr_res.get("rec_texts"):
            return "\n".join(str(text) for text in ocr_res["rec_texts"] if str(text).strip())

    if isinstance(result, dict):
        rec_texts = result.get("rec_texts") or result.get("text")
        if isinstance(rec_texts, list):
            return "\n".join(str(text) for text in rec_texts if str(text).strip())
        if isinstance(rec_texts, str):
            return rec_texts

    return str(result)


# Plain OCR mode is simpler than structure mode: read page text and label each page.
def paddle_text_ocr_to_raw_markdown(src: Path, args) -> str:
    ocr = get_text_ocr(args)
    output = ocr.predict(input=str(src))
    parts = [f"# {src.stem}"]
    for page_num, result in enumerate(output, 1):
        text = strip_html_noise(extract_text_from_paddle_result(result))
        if text:
            parts.append(f"## Страница {page_num}\n\n{text}")
    return "\n\n".join(parts)


def convert_file_to_raw_markdown(src: Path, args, office_config: dict) -> tuple[str, str]:
    ext = src.suffix.lower()
    if ext in PADDLE_EXTENSIONS:
        return paddle_to_raw_markdown(src, args), "paddle-ppstructurev3"
    raw, mode = convert_to_raw_markdown(src, office_config)
    return raw, mode


# One source file becomes one cleaned .md file, using Paddle for images/scans and
# the base converter for Office-style inputs.
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
            raise ValueError("Resulting markdown is empty after PaddleOCR/cleaning")

        out_file.write_text(cleaned, encoding="utf-8")
        return str(src), "ok", mode, stats
    except Exception as exc:
        return str(src), "fail", f"{type(exc).__name__}: {exc}", empty_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert documents to clean RAG Markdown using PaddleOCR PP-StructureV3."
    )
    parser.add_argument("--input", "-i", required=True, help="Input folder with source files.")
    parser.add_argument("--output", "-o", required=True, help="Output folder for .md files.")
    parser.add_argument("--failed-log", help="Path for failed conversions log.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .md files.")
    parser.add_argument("--libreoffice", help="Path to LibreOffice soffice executable for .doc/.rtf.")
    parser.add_argument(
        "--lang",
        default="ru",
        help="PaddleOCR recognition language. Default: ru. Use en for English-only documents.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="PaddleOCR inference device. Default: gpu. Use cpu if GPU/CUDA is unstable.",
    )
    parser.add_argument(
        "--paddle-mode",
        choices=["structure", "ocr"],
        default="structure",
        help="structure uses PP-StructureV3; ocr uses plain PaddleOCR text recognition.",
    )

    parser.add_argument(
        "--use-doc-orientation-classify",
        action="store_true",
        help="Enable PaddleOCR document orientation classification.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action="store_true",
        help="Enable PaddleOCR document unwarping for warped scans/photos.",
    )
    parser.add_argument(
        "--use-textline-orientation",
        action="store_true",
        help="Enable PaddleOCR text line orientation detection.",
    )
    parser.add_argument(
        "--no-table-recognition",
        dest="use_table_recognition",
        action="store_false",
        help="Disable PaddleOCR table recognition.",
    )
    parser.add_argument(
        "--use-seal-recognition",
        action="store_true",
        help="Enable PaddleOCR seal recognition.",
    )
    parser.add_argument(
        "--no-formula-recognition",
        dest="use_formula_recognition",
        action="store_false",
        help="Disable PaddleOCR formula recognition.",
    )
    parser.set_defaults(use_table_recognition=True, use_formula_recognition=True)
    return parser.parse_args()


# Command-line workflow: collect files, configure Paddle if needed, process each file,
# then print conversion and cleanup statistics.
def main():
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    failed_log = Path(args.failed_log).expanduser().resolve() if args.failed_log else output_dir / "failed_paddle_convert.txt"

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
    print(f"Files: {len(files)} | OCR: PaddleOCR {args.paddle_mode} | workers: 1")
    if not files:
        print(f"Output: {output_dir}")
        print("No supported files found.")
        sys.exit(0)

    if any(file.suffix.lower() in PADDLE_EXTENSIONS for file in files) and args.paddle_mode == "structure":
        configure_paddle_pipeline(args)

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
