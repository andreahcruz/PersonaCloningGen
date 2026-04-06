FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate.py user_interface.py ./
COPY formats/ ./formats/

ENV CHROMA_USE_HTTP=true
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV CHROMA_COLLECTION_NAME=lemkin_content

EXPOSE 8501

CMD ["streamlit", "run", "user_interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
