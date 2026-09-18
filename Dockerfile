FROM python:3.11-slim

WORKDIR /app

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY generate.py user_interface.py ./
COPY formats/ ./formats/
# generate.py imports these two modules from spark_jobs/. docker-compose bind-mounts the whole
# folder over this, but a standalone image (cloud deploy) would crash on import without them.
COPY spark_jobs/corpus_footer_scrub.py spark_jobs/output_checks.py ./spark_jobs/

ENV CHROMA_USE_HTTP=true
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV CHROMA_COLLECTION_NAME=lemkin_content

EXPOSE 8501

CMD ["streamlit", "run", "user_interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
