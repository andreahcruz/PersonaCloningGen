"""
Airflow DAG — Paul Graham Essay ETL Pipeline

Task 1: extract_to_minio      — HuggingFace → MinIO (pg-raw)
Task 2: trigger_spark_clean    — Spark job: clean, chunk, embed → MinIO (pg-processed)
Task 3: load_to_snowflake      — pg-processed → Snowflake table
Task 4: load_to_chroma         — pg-processed embeddings → Chroma collection
Task 5: extract_persona        — pg-raw → stylometric analysis → persona_profile.json → pg-processed
"""
import json
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


def _minio_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )


# ── Task 1 ───────────────────────────────────────────────────────────────────

def extract_to_minio():
    from datasets import load_dataset

    ds = load_dataset("aadi-blogs/paul_graham_essays", split="train")
    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_RAW", "pg-raw")

    for i, row in enumerate(ds):
        body = json.dumps(dict(row), default=str)
        s3.put_object(Bucket=bucket, Key=f"essay_{i}.json", Body=body)

    print(f"Uploaded {len(ds)} essays to MinIO bucket '{bucket}'")


# ── Task 3 ───────────────────────────────────────────────────────────────────

def load_to_snowflake():
    import snowflake.connector

    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_PROCESSED", "pg-processed")

    objects = s3.list_objects_v2(Bucket=bucket, Prefix="chunks/")
    rows = []
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json") or key.startswith("chunks/_"):
            continue
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        for line in data.strip().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS PG_ESSAYS_CLEANED (
            essay_id       INTEGER,
            chunk_index    INTEGER,
            title          VARCHAR(500),
            date_original  VARCHAR(100),
            date_ts        TIMESTAMP_NTZ,
            url            VARCHAR(1000),
            chunk_text     VARCHAR(16777216),
            word_count     INTEGER
        )
    """)
    cur.execute("TRUNCATE TABLE IF EXISTS PG_ESSAYS_CLEANED")

    insert_sql = """
        INSERT INTO PG_ESSAYS_CLEANED
            (essay_id, chunk_index, title, date_original, date_ts, url, chunk_text, word_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    batch = []
    for r in rows:
        batch.append((
            r.get("essay_id"),
            r.get("chunk_index"),
            r.get("title"),
            r.get("date"),
            r.get("date_ts"),
            r.get("url"),
            r.get("chunk_text"),
            r.get("word_count"),
        ))

    if batch:
        cur.executemany(insert_sql, batch)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Loaded {len(batch)} rows into Snowflake PG_ESSAYS_CLEANED")


# ── Task 4 ───────────────────────────────────────────────────────────────────

def load_to_chroma():
    import chromadb

    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_PROCESSED", "pg-processed")

    objects = s3.list_objects_v2(Bucket=bucket, Prefix="chunks/")
    rows = []
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json") or key.startswith("chunks/_"):
            continue
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        for line in data.strip().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    chroma_host = os.environ.get("CHROMA_HOST", "chroma")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection("pg_essays")

    batch_size = 100
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for r in chunk:
            emb = r.get("embedding")
            if not emb or not isinstance(emb, list) or len(emb) == 0:
                continue

            doc_id = f"essay_{r.get('essay_id', 0)}_chunk_{r.get('chunk_index', 0)}"
            ids.append(doc_id)
            documents.append(r.get("chunk_text", ""))
            embeddings.append(emb)
            metadatas.append({
                "essay_id": r.get("essay_id", 0),
                "title": r.get("title", ""),
                "chunk_index": r.get("chunk_index", 0),
                "date": r.get("date", ""),
                "url": r.get("url", ""),
                "word_count": r.get("word_count", 0),
            })

        if ids:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    print(f"Upserted {len(rows)} chunks into Chroma collection 'pg_essays'")


# ── Task 5 ───────────────────────────────────────────────────────────────────

def extract_persona():
    from collections import Counter

    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    s3 = _minio_client()
    bucket_raw = os.environ.get("MINIO_BUCKET_RAW", "pg-raw")
    bucket_processed = os.environ.get("MINIO_BUCKET_PROCESSED", "pg-processed")

    objects = s3.list_objects_v2(Bucket=bucket_raw)
    all_text = []
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        data = s3.get_object(Bucket=bucket_raw, Key=key)["Body"].read().decode()
        essay = json.loads(data)
        text = essay.get("text", "")
        if text:
            all_text.append(text)

    stop_words = set(stopwords.words("english"))
    all_words = []
    all_sentences = []
    for text in all_text:
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)
        words = word_tokenize(text.lower())
        content_words = [
            w for w in words
            if w.isalpha() and w not in stop_words and len(w) > 2
        ]
        all_words.extend(content_words)

    word_freq = Counter(all_words)
    top_50_words = [w for w, _ in word_freq.most_common(50)]

    starter_bigrams = Counter()
    for sent in all_sentences:
        words = sent.split()[:2]
        if len(words) == 2:
            bigram = " ".join(w.lower() for w in words)
            starter_bigrams[bigram] += 1
    common_starters = [b for b, _ in starter_bigrams.most_common(20)]

    FRAMING_PATTERNS = [
        "the thing is", "in other words", "the problem is",
        "the point is", "what i mean is", "the question is",
        "the reason is", "the trick is", "the way to",
        "one of the", "it turns out", "the key is",
        "what matters is", "the real", "in practice",
        "in fact", "for example", "in general",
        "the most important", "it seems like",
    ]
    framing_counts = Counter()
    lower_corpus = " ".join(all_text).lower()
    for phrase in FRAMING_PATTERNS:
        count = lower_corpus.count(phrase)
        if count > 0:
            framing_counts[phrase] = count
    top_framing = [p for p, _ in framing_counts.most_common(15)]

    bigram_counter = Counter()
    for text in all_text:
        words = word_tokenize(text.lower())
        content = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
        for i in range(len(content) - 1):
            bigram_counter[f"{content[i]} {content[i+1]}"] += 1
    theme_words = [w for w, _ in word_freq.most_common(20)]
    theme_bigrams = [b for b, _ in bigram_counter.most_common(20)]
    top_themes = theme_words + theme_bigrams

    persona_profile = {
        "vocabulary_tendencies": {
            "top_50_content_words": top_50_words,
        },
        "structure_patterns": {
            "common_sentence_starters_bigrams": common_starters,
        },
        "favorite_framing": {
            "favorite_framing_phrases": top_framing,
        },
        "favorite_themes": {
            "top_content_words_and_bigrams": top_themes,
        },
        "meta": {
            "source": "Paul Graham essays (HuggingFace)",
            "num_essays": len(all_text),
            "num_sentences": len(all_sentences),
            "num_content_words": len(all_words),
            "generated_at": datetime.utcnow().isoformat(),
        },
    }

    s3.put_object(
        Bucket=bucket_processed,
        Key="persona_profile.json",
        Body=json.dumps(persona_profile, indent=2),
    )
    print(f"Wrote persona_profile.json to MinIO bucket '{bucket_processed}'")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="pg_essay_pipeline",
    description="Paul Graham Essay ETL: Extract → Spark Transform → Snowflake + Chroma + Persona",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["etl", "paul-graham", "persona"],
) as dag:

    task_extract = PythonOperator(
        task_id="extract_to_minio",
        python_callable=extract_to_minio,
    )

    task_spark = SparkSubmitOperator(
        task_id="trigger_spark_clean",
        application="/opt/airflow/spark_jobs/clean_and_embed.py",
        conn_id="spark_default",
        conf={
            "spark.hadoop.fs.s3a.endpoint": "{{ var.value.get('MINIO_ENDPOINT', 'http://minio:9000') }}",
            "spark.hadoop.fs.s3a.access.key": "minioadmin",
            "spark.hadoop.fs.s3a.secret.key": "minioadmin",
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        },
        env_vars={
            "MINIO_ENDPOINT": os.environ.get("MINIO_ENDPOINT", "http://minio:9000"),
            "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            "MINIO_BUCKET_RAW": os.environ.get("MINIO_BUCKET_RAW", "pg-raw"),
            "MINIO_BUCKET_PROCESSED": os.environ.get("MINIO_BUCKET_PROCESSED", "pg-processed"),
            "OLLAMA_BASE": os.environ.get("OLLAMA_BASE", "http://host.docker.internal:11434"),
        },
        verbose=True,
    )

    task_snowflake = PythonOperator(
        task_id="load_to_snowflake",
        python_callable=load_to_snowflake,
    )

    task_chroma = PythonOperator(
        task_id="load_to_chroma",
        python_callable=load_to_chroma,
    )

    task_persona = PythonOperator(
        task_id="extract_persona",
        python_callable=extract_persona,
    )

    task_extract >> task_spark >> [task_snowflake, task_chroma]
    task_extract >> task_persona
