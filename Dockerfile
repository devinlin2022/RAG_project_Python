FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=7860

WORKDIR /app

# pdfminer/pdfplumber are pure Python; pypdfium2 ships manylinux wheels.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Documents, the SQLite database and the index all live here.
VOLUME ["/app/data"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["python", "run.py"]
