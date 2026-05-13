# Document to Markdown Web App

This project wraps the conversion scripts in a small FastAPI web app.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Railway

Railway can run the app with:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

`main.py` exports the FastAPI app for Railway and other platforms that expect
that default entry point.

## PaddleOCR mode

PaddleOCR dependencies are included in `requirements.txt` so Railway and local
setups use one dependency list. Use `PaddleOCR` in the web form for scanned PDFs
and image files.

For local testing, choose `OCR text` first. `Structure` mode is heavier because
it loads extra layout, table, and formula models.

## Files

- `app.py`: FastAPI web wrapper.
- `templates/index.html`: Upload/download page.
- `prepare_rag_markdown.py`: Standard PDF/Office/Markdown converter.
- `prepare_rag_markdown_paddle.py`: PaddleOCR converter for scans/images.
