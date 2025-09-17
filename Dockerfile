FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_MANIFEST_PATH="/data/artifact_manifest.json"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
