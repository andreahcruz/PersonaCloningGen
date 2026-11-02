FROM python:3.11-slim

WORKDIR /app

COPY lightrag_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY lightrag_service/ ./lightrag_service/

EXPOSE 8010

CMD ["uvicorn", "lightrag_service.main:app", "--host", "0.0.0.0", "--port", "8010"]
