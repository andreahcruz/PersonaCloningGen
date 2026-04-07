"""
Airflow DAG — Jason Lemkin content ETL

Task 1: extract_to_minio      — local JSONL (blog, LinkedIn, YouTube, X) → MinIO (raw/)
Task 2: trigger_spark_clean   — Spark: clean, chunk, embed → MinIO (chunks/)
Task 3: load_to_chroma        — chunks → Chroma
Task 4: extract_persona       — raw JSON → persona_profile → MinIO + host data/
"""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def _normalize_blog(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = (row.get("content") or row.get("text") or "").strip()
    if not text:
        return None
    title = (row.get("title") or "Blog post")[:500]
    return {
        "title": title,
        "date": row.get("date") or "",
        "url": row.get("url") or "",
        "text": text,
        "source": "blog",
    }


def _normalize_linkedin(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = (row.get("content") or "").strip()
    if not text:
        return None
    short = text[:120] + ("..." if len(text) > 120 else "")
    return {
        "title": short,
        "date": row.get("published_text") or "",
        "url": "",
        "text": text,
        "source": "linkedin",
    }


def _normalize_youtube(row: Dict[str, Any], source_tag: str) -> Optional[Dict[str, Any]]:
    text = (row.get("transcript_text") or "").strip()
    if not text:
        return None
    date = row.get("upload_date_iso") or ""
    if not date and row.get("fetched_at_utc"):
        date = str(row["fetched_at_utc"])[:10]
    return {
        "title": (row.get("video_title") or "YouTube video")[:500],
        "date": date,
        "url": row.get("video_url") or "",
        "text": text,
        "source": source_tag,
    }


def _normalize_x(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """X/Twitter timeline export: skip replies, reposts/quotes, and very short stubs."""
    if row.get("is_reply") or row.get("is_repost_or_quote"):
        return None
    text = (row.get("text") or "").strip()
    if len(text) < 25:
        return None
    preview = text[:120] + ("..." if len(text) > 120 else "")
    return {
        "title": preview,
        "date": row.get("created_at") or "",
        "url": row.get("url") or "",
        "text": text,
        "source": "x",
    }


def extract_to_minio():
    """Read Lemkin JSONL files from LEMKIN_DATA_DIR and upload normalized JSON to MinIO raw/."""
    data_dir = Path(os.environ.get("LEMKIN_DATA_DIR", "/opt/airflow/data"))
    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_RAW", "lemkin-raw")
    raw_prefix = "raw"

    loaders: List[Tuple[str, Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]]] = [
        ("jasonlemkin_blog.jsonl", lambda r: _normalize_blog(r)),
        ("jasonlemkinlinkedin.jsonl", lambda r: _normalize_linkedin(r)),
        ("jasonmlemkinyoutubetranscripts.jsonl", lambda r: _normalize_youtube(r, "youtube_jason")),
        ("saastryoutubetranscripts.jsonl", lambda r: _normalize_youtube(r, "youtube_saastr")),
        ("jasonlk_originals.jsonl", lambda r: _normalize_x(r)),
    ]

    total = 0
    for filename, normalizer in loaders:
        path = data_dir / filename
        if not path.is_file():
            print(f"Skip missing file: {path}")
            continue
        idx = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skip bad JSON in {filename}: {e}")
                    continue
                rec = normalizer(row)
                if not rec:
                    continue
                key = f"{raw_prefix}/{Path(filename).stem}_{idx:06d}.json"
                body = json.dumps(rec, ensure_ascii=False)
                s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
                idx += 1
                total += 1
        print(f"Uploaded {idx} records from {filename}")

    if total == 0:
        raise RuntimeError(
            f"No documents uploaded from {data_dir}. "
            "Add at least one of: jasonlemkin_blog.jsonl, jasonlemkinlinkedin.jsonl, "
            "jasonmlemkinyoutubetranscripts.jsonl, saastryoutubetranscripts.jsonl, "
            "jasonlk_originals.jsonl under ./data (mounted in the container)."
        )
    print(f"Uploaded {total} total documents to s3://{bucket}/{raw_prefix}/")


def load_to_chroma():
    import chromadb

    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")

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
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

    tenant = os.environ.get("CHROMA_TENANT", DEFAULT_TENANT)
    database = os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE)
    client = chromadb.HttpClient(
        host=chroma_host, port=chroma_port, tenant=tenant, database=database
    )
    collection = client.get_or_create_collection(collection_name)

    batch_size = 100
    upserted = 0
    skipped_no_embedding = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for r in chunk:
            emb = r.get("embedding")
            if not emb or not isinstance(emb, list) or len(emb) == 0:
                skipped_no_embedding += 1
                continue

            src = r.get("source") or "unknown"
            doc_id = f"{src}_{r.get('essay_id', 0)}_{r.get('chunk_index', 0)}"
            ids.append(doc_id)
            documents.append(r.get("chunk_text", ""))
            embeddings.append(emb)
            metadatas.append({
                "essay_id": r.get("essay_id", 0),
                "source": src,
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
            upserted += len(ids)

    print(
        f"Chroma collection '{collection_name}': upserted {upserted} rows with embeddings; "
        f"skipped {skipped_no_embedding} JSON lines with missing/empty embedding "
        f"(out of {len(rows)} lines read from MinIO)."
    )


def extract_persona():
    from collections import Counter

    import nltk
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    s3 = _minio_client()
    bucket_raw = os.environ.get("MINIO_BUCKET_RAW", "lemkin-raw")
    bucket_processed = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")

    objects = s3.list_objects_v2(Bucket=bucket_raw, Prefix="raw/")
    all_text = []
    for obj in objects.get("Contents", []):
        key = obj["Key"]
        if not key.endswith(".json"):
            continue
        data = s3.get_object(Bucket=bucket_raw, Key=key)["Body"].read().decode()
        rec = json.loads(data)
        text = rec.get("text", "")
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
        "the thing is", "in other words", "the bottom line",
        "here is", "here's what", "the reality is",
        "what matters is", "at the end of the day",
        "in fact", "for example", "in general",
        "if you", "when you", "the key is",
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
            "source": "Jason Lemkin — blog, LinkedIn, X posts, SaaStr / channel YouTube transcripts",
            "num_documents": len(all_text),
            "num_sentences": len(all_sentences),
            "num_content_words": len(all_words),
            "generated_at": datetime.utcnow().isoformat(),
        },
    }

    body = json.dumps(persona_profile, indent=2)
    s3.put_object(
        Bucket=bucket_processed,
        Key="persona_profile.json",
        Body=body,
    )
    local_path = os.environ.get("PERSONA_LOCAL_PATH", "/opt/airflow/data/persona_profile.json")
    try:
        Path(local_path).write_text(body, encoding="utf-8")
        print(f"Also wrote persona to {local_path} (host ./data when mounted)")
    except OSError as e:
        print(f"Could not write local persona file: {e}")

    print(f"Wrote persona_profile.json to MinIO bucket '{bucket_processed}'")


with DAG(
    dag_id="lemkin_content_pipeline",
    description="Lemkin ETL: local JSONL → MinIO → Spark → Chroma + Persona",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["etl", "lemkin", "persona", "jason-lemkin"],
) as dag:

    task_extract = PythonOperator(
        task_id="extract_to_minio",
        python_callable=extract_to_minio,
    )

    task_spark = SparkSubmitOperator(
        task_id="trigger_spark_clean",
        application="/opt/airflow/spark_jobs/clean_and_embed.py",
        conn_id="spark_default",
        # PySpark in the Airflow image does not ship S3A jars; MinIO/s3a needs these on the classpath
        packages="org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
        conf={
            "spark.driver.memory": "2g",
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
            "MINIO_BUCKET_RAW": os.environ.get("MINIO_BUCKET_RAW", "lemkin-raw"),
            "MINIO_BUCKET_PROCESSED": os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed"),
            "OLLAMA_BASE": os.environ.get("OLLAMA_BASE", "http://host.docker.internal:11434"),
            # Fewer partitions = fewer parallel Ollama calls (see spark_jobs/clean_and_embed.py)
            "SPARK_EMBED_PARTITIONS": os.environ.get("SPARK_EMBED_PARTITIONS", "1"),
            "OLLAMA_EMBED_MAX_RETRIES": os.environ.get("OLLAMA_EMBED_MAX_RETRIES", "5"),
            "OLLAMA_EMBED_DELAY_SEC": os.environ.get("OLLAMA_EMBED_DELAY_SEC", "0.03"),
        },
        # Long embed + write; Airflow default is no cap, but this documents intent and avoids surprises if a cap is set globally.
        execution_timeout=timedelta(hours=12),
        verbose=True,
    )

    task_chroma = PythonOperator(
        task_id="load_to_chroma",
        python_callable=load_to_chroma,
    )

    task_persona = PythonOperator(
        task_id="extract_persona",
        python_callable=extract_persona,
    )

    task_extract >> task_spark >> task_chroma
    task_extract >> task_persona
