FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt

COPY generate.py ./
COPY spark_jobs/corpus_footer_scrub.py ./spark_jobs/corpus_footer_scrub.py
COPY spark_jobs/output_checks.py ./spark_jobs/output_checks.py
COPY formats/ ./formats/
COPY scripts/capture_personarag_smoke.py ./scripts/capture_personarag_smoke.py
COPY backend/main.py ./main.py

ENV CHROMA_USE_HTTP=true
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV CHROMA_COLLECTION_NAME=lemkin_content

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
