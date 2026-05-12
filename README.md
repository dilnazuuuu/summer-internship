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
uvicorn app:app --host 0.0.0.0 --port $PORT
```

`main.py` also exports the same app as `main:app` for platforms that expect that default.

## PaddleOCR mode

Install the optional PaddleOCR packages if you want to convert scans and images:

```bash
python -m pip install -r requirements-paddle.txt
```

Use `PaddleOCR` in the web form for scanned PDFs and image files.

For local testing, choose `OCR text` first. `Structure` mode is heavier because
it loads extra layout, table, and formula models.

## Files

- `app.py`: FastAPI web wrapper.
- `templates/index.html`: Upload/download page.
- `prepare_rag_markdown.py`: Standard PDF/Office/Markdown converter.
- `prepare_rag_markdown_paddle.py`: PaddleOCR converter for scans/images.
