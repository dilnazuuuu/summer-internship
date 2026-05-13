from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock, Thread

# Keep Paddle in stable CPU mode on Railway before any OCR-related imports.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

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
JOB_DIR = Path(os.environ.get("RAG_MD_JOB_DIR", "/tmp/rag_md_jobs"))

app = FastAPI(title="Document to Markdown Converter")
jobs_lock = Lock()


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "uploaded_file").name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "uploaded_file"


def cleanup_folder(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def job_path(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    return JOB_DIR / f"{job_id}.json"


def read_job(job_id: str) -> dict | None:
    path = job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_job(job_id: str, job: dict) -> None:
    path = job_path(job_id)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(job), encoding="utf-8")
    temp_path.replace(path)


def cleanup_job(job_id: str) -> None:
    with jobs_lock:
        job = read_job(job_id)
        if job:
            job_path(job_id).unlink(missing_ok=True)
    if job and job.get("work_dir"):
        cleanup_folder(job["work_dir"])


def cleanup_job_later(job_id: str, delay_seconds: int = 600) -> None:
    time.sleep(delay_seconds)
    cleanup_job(job_id)


def cleanup_job_files(job_id: str) -> None:
    with jobs_lock:
        job = read_job(job_id)
        work_dir = job.get("work_dir") if job else None
        if job:
            job["work_dir"] = None
            write_job(job_id, job)
    if work_dir:
        cleanup_folder(work_dir)


def update_job(job_id: str, **updates) -> None:
    with jobs_lock:
        job = read_job(job_id)
        if job:
            job.update(updates)
            write_job(job_id, job)


def update_job_progress(job_id: str, current_page: int, total_pages: int) -> None:
    percent = 0
    if total_pages > 0:
        percent = round((current_page / total_pages) * 100)
    update_job(
        job_id,
        current_page=current_page,
        total_pages=total_pages,
        progress_percent=max(0, min(100, percent)),
    )


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
        raise HTTPException(status_code=400, detail="Invalid OCR mode.")
    if device not in {"cpu", "gpu"}:
        raise HTTPException(status_code=400, detail="Invalid OCR device.")
    if lang not in PADDLE_LANGUAGES:
        raise HTTPException(status_code=400, detail="Invalid OCR language.")

    # Force lightweight OCR mode for hosted deploys; structure mode is too heavy
    # for the current Railway setup.
    paddle_mode = "ocr"

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
        use_formula_recognition=False,
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


def run_conversion(job_id, input_file, input_dir, output_dir, engine, args, office_config):
    try:
        if engine == "standard":
            config = build_standard_config(input_file)
            src, status, message, _stats = process_standard_file(
                str(input_file),
                str(input_dir),
                str(output_dir),
                config,
            )
        else:
            def progress_callback(current_page: int, total_pages: int) -> None:
                update_job_progress(job_id, current_page, total_pages)

            if input_file.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
                try:
                    if args.paddle_mode == "structure":
                        configure_paddle_pipeline(args)
                except RuntimeError as exc:
                    update_job(job_id, status="failed", error=str(exc))
                    cleanup_job_files(job_id)
                    return

            src, status, message, _stats = process_paddle_file(
                input_file,
                input_dir,
                output_dir,
                args,
                office_config,
                progress_callback=progress_callback,
            )

        if status != "ok":
            update_job(job_id, status="failed", error=f"Conversion failed for {src}: {message}")
            cleanup_job_files(job_id)
            return

        markdown_file = output_dir / f"{input_file.stem}.md"
        if not markdown_file.exists():
            update_job(job_id, status="failed", error="Conversion finished but no Markdown file was created.")
            cleanup_job_files(job_id)
            return

        job = read_job(job_id) or {}
        done_updates = {"status": "done", "file": str(markdown_file), "progress_percent": 100}
        total_pages = job.get("total_pages")
        if total_pages:
            done_updates["current_page"] = total_pages
        update_job(job_id, **done_updates)
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc))
        cleanup_job_files(job_id)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


@app.post("/convert")
async def convert_document(
    file: UploadFile = File(...),
    engine: str = Form("standard"),
    paddle_mode: str = Form("ocr"),
    lang: str = Form("ru"),
    device: str = Form("cpu"),
):
    job_id = str(uuid.uuid4())
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

        args = None
        office_config = None
        if engine == "paddle":
            args = build_paddle_args(paddle_mode, lang, device)
            office_config = build_paddle_office_config(input_file, args)
    except Exception:
        cleanup_folder(str(work_dir))
        raise

    with jobs_lock:
        write_job(
            job_id,
            {
                "job_id": job_id,
                "status": "processing",
                "work_dir": str(work_dir),
                "file": None,
                "error": None,
                "filename": filename,
                "current_page": 0,
                "total_pages": None,
                "progress_percent": 0,
            },
        )

    Thread(
        target=run_conversion,
        args=(job_id, input_file, input_dir, output_dir, engine, args, office_config),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "status": "processing",
        "status_url": f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }


@app.get("/status/{job_id}")
def get_status(job_id: str):
    with jobs_lock:
        job = read_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "error": job.get("error"),
            "current_page": job.get("current_page"),
            "total_pages": job.get("total_pages"),
            "progress_percent": job.get("progress_percent"),
            "download_url": f"/download/{job_id}" if job.get("status") == "done" else None,
        }


@app.get("/download/{job_id}")
def download_file(job_id: str, background_tasks: BackgroundTasks):
    with jobs_lock:
        job = read_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("status") != "done":
            raise HTTPException(status_code=400, detail="File not ready")
        output_file = job.get("file")

    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Converted file is no longer available")

    background_tasks.add_task(cleanup_job_later, job_id)
    return FileResponse(
        output_file,
        media_type="text/markdown",
        filename="result.md",
        background=background_tasks,
    )
