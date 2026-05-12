from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

# Keep PaddleOCR/PaddleX cache in a writable temp folder for local and hosted runs.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/private/tmp/paddlex-cache")

from prepare_rag_markdown import (
    EXTENSIONS as STANDARD_EXTENSIONS,
    process_file as process_standard_file,
    validate_args as validate_standard_args,
)
from prepare_rag_markdown_paddle import (
    EXTENSIONS as PADDLE_EXTENSIONS,
    OFFICE_EXTENSIONS,
    configure_paddle_pipeline,
    process_file as process_paddle_file,
    validate_office_args,
)


# Small web wrapper around the conversion scripts.
# Users upload one document, the app converts it in a temporary folder, then
# returns the generated Markdown file as a download.

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "templates" / "index.html"
PADDLE_LANGUAGES = {"ru", "kk", "en"}

app = FastAPI(title="Document to Markdown Converter")


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "uploaded_file").name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "uploaded_file"


def cleanup_folder(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def build_standard_config(input_file: Path) -> dict:
    args = argparse.Namespace(
        overwrite=True,
        no_ocr=True,
        poppler_path=None,
        tesseract_cmd=None,
        libreoffice=None,
    )
    try:
        return validate_standard_args(args, [input_file])
    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail="Standard converter dependencies are missing. Check requirements.txt.",
        ) from exc


def build_paddle_args(
    paddle_mode: str,
    lang: str,
    device: str,
) -> argparse.Namespace:
    if paddle_mode not in {"structure", "ocr"}:
        raise HTTPException(status_code=400, detail="Invalid PaddleOCR mode.")
    if device not in {"cpu", "gpu"}:
        raise HTTPException(status_code=400, detail="Invalid OCR device.")
    if lang not in PADDLE_LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid OCR language.")

    return argparse.Namespace(
        overwrite=True,
        libreoffice=None,
        lang=lang,
        device=device,
        paddle_mode=paddle_mode,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_seal_recognition=False,
        use_formula_recognition=True,
    )


def build_paddle_office_config(input_file: Path, args: argparse.Namespace) -> dict:
    office_files = []
    if input_file.suffix.lower() in OFFICE_EXTENSIONS - {".pdf"}:
        office_files.append(input_file)

    office_args = argparse.Namespace(
        overwrite=True,
        no_ocr=True,
        poppler_path=None,
        tesseract_cmd=None,
        libreoffice=args.libreoffice,
    )
    try:
        return validate_office_args(office_args, office_files)
    except SystemExit as exc:
        raise HTTPException(
            status_code=500,
            detail="Office converter dependencies are missing. Check requirements.txt.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.post("/convert")
async def convert_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form("standard"),
    paddle_mode: str = Form("ocr"),
    lang: str = Form("ru"),
    device: str = Form("cpu"),
):
    filename = safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()

    if engine == "standard":
        supported_extensions = STANDARD_EXTENSIONS
    elif engine == "paddle":
        supported_extensions = PADDLE_EXTENSIONS
    else:
        raise HTTPException(status_code=400, detail="Invalid conversion engine.")

    if suffix not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported}",
        )

    work_dir = Path(tempfile.mkdtemp(prefix="rag_md_web_"))
    try:
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_file = input_dir / filename
        with input_file.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        if engine == "standard":
            config = build_standard_config(input_file)
            src, status, message, _stats = process_standard_file(
                str(input_file),
                str(input_dir),
                str(output_dir),
                config,
            )
        else:
            args = build_paddle_args(paddle_mode, lang, device)
            office_config = build_paddle_office_config(input_file, args)
            if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                try:
                    if args.paddle_mode == "structure":
                        configure_paddle_pipeline(args)
                except RuntimeError as exc:
                    raise HTTPException(status_code=500, detail=str(exc)) from exc
            src, status, message, _stats = process_paddle_file(
                input_file,
                input_dir,
                output_dir,
                args,
                office_config,
            )

        if status != "ok":
            raise HTTPException(status_code=500, detail=f"Conversion failed for {src}: {message}")

        markdown_file = output_dir / f"{input_file.stem}.md"
        if not markdown_file.exists():
            raise HTTPException(status_code=500, detail="Conversion finished but no Markdown file was created.")
    except Exception:
        cleanup_folder(str(work_dir))
        raise

    background_tasks.add_task(cleanup_folder, str(work_dir))

    return FileResponse(
        markdown_file,
        media_type="text/markdown",
        filename=f"{input_file.stem}.md",
        background=background_tasks,
    )
