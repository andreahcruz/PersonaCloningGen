"""
Airflow DAG — Jason Lemkin content ETL

Task 1: extract_to_minio       — local JSONL (blog, LinkedIn, YouTube, X) → MinIO (raw/)
Task 2: trigger_spark_clean    — Spark: clean, chunk, embed → MinIO (chunks/) + training/
Task 3: load_to_chroma         — chunks → Chroma
Task 4: extract_persona        — raw JSON → persona_profile → MinIO + host data/
Task 5: mark_training_ready    — promote Spark's part-*.txt to a stable
                                 training/dataset.jsonl + write _READY sentinel
                                 so the host-side fine-tune watcher can pick it up.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# Airflow attaches its per-task file handler to this logger, so everything logged here
# shows up in the task log (UI → task → Log).
log = logging.getLogger("airflow.task")

# load_to_chroma (and the Spark job, before it writes) fail when more than this share of
# chunk rows has no embedding — otherwise a dead Ollama silently yields a half-empty index.
MAX_MISSING_EMBEDDING_FRACTION = float(os.environ.get("MAX_MISSING_EMBEDDING_FRACTION", "0.10"))


def _require_env(*names: str) -> Dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "They are set for Airflow in docker-compose.yml (x-airflow-common)."
        )
    return {n: os.environ[n] for n in names}


def _minio_client():
    import boto3
    from botocore.config import Config

    env = _require_env("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")
    return boto3.client(
        "s3",
        endpoint_url=env["MINIO_ENDPOINT"],
        aws_access_key_id=env["MINIO_ACCESS_KEY"],
        aws_secret_access_key=env["MINIO_SECRET_KEY"],
        # Retry transient MinIO errors (restart, throttling) instead of failing the task at once.
        config=Config(
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def _check_minio(s3, *buckets: str) -> None:
    """Fail fast, with a fix hint, if MinIO or a bucket cannot be reached."""
    for bucket in buckets:
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception as e:  # noqa: BLE001 - re-raised below with context
            raise RuntimeError(
                f"MinIO bucket '{bucket}' is not reachable at {os.environ.get('MINIO_ENDPOINT')}: {e}. "
                "Check `docker compose ps minio`; the buckets are created by the minio-setup service."
            ) from e


def _list_keys(s3, bucket: str, prefix: str) -> List[str]:
    """Every key under ``prefix``. list_objects_v2 returns at most 1000 keys per call, so page."""
    keys: List[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def _has_embedding(row: Dict[str, Any]) -> bool:
    emb = row.get("embedding")
    return isinstance(emb, list) and len(emb) > 0


def _check_embedding_coverage(total: int, missing: int) -> None:
    """Raise if too many chunk rows lack an embedding (would be silent data loss)."""
    if total == 0:
        raise RuntimeError("No chunk rows to load.")
    fraction = missing / total
    if missing:
        log.warning(
            "%d of %d chunk rows (%.1f%%) have no embedding (limit %.0f%%).",
            missing, total, fraction * 100, MAX_MISSING_EMBEDDING_FRACTION * 100,
        )
    if missing == total or fraction > MAX_MISSING_EMBEDDING_FRACTION:
        raise RuntimeError(
            f"{missing} of {total} chunk rows have no embedding, above the "
            f"{MAX_MISSING_EMBEDDING_FRACTION:.0%} limit (MAX_MISSING_EMBEDDING_FRACTION). "
            "Ollama was probably unreachable or overloaded during trigger_spark_clean: "
            "check `ollama serve`, lower SPARK_EMBED_PARTITIONS, then re-run trigger_spark_clean."
        )


def _log_task_failure(context: Dict[str, Any]) -> None:
    ti = context.get("task_instance")
    log.error(
        "TASK FAILED dag=%s task=%s run=%s try=%s exception=%r log=%s",
        getattr(ti, "dag_id", "?"), getattr(ti, "task_id", "?"), context.get("run_id", "?"),
        getattr(ti, "try_number", "?"), context.get("exception"), getattr(ti, "log_url", "?"),
    )


def _log_task_retry(context: Dict[str, Any]) -> None:
    ti = context.get("task_instance")
    log.warning(
        "TASK WILL RETRY dag=%s task=%s run=%s try=%s exception=%r",
        getattr(ti, "dag_id", "?"), getattr(ti, "task_id", "?"), context.get("run_id", "?"),
        getattr(ti, "try_number", "?"), context.get("exception"),
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
    _check_minio(s3, bucket)

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
            log.warning("Skip missing file: %s", path)
            continue
        idx = 0
        filtered = 0
        bad_json = 0
        try:
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        bad_json += 1
                        log.warning("Skip bad JSON in %s line %d: %s", filename, line_no, e)
                        continue
                    if not isinstance(row, dict):
                        bad_json += 1
                        log.warning("Skip non-object JSON in %s line %d", filename, line_no)
                        continue
                    rec = normalizer(row)
                    if not rec:
                        filtered += 1
                        continue
                    key = f"{raw_prefix}/{Path(filename).stem}_{idx:06d}.json"
                    body = json.dumps(rec, ensure_ascii=False)
                    s3.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))
                    idx += 1
                    total += 1
        except UnicodeDecodeError as e:
            raise RuntimeError(
                f"{path} is not valid UTF-8 ({e}). Re-export the file as UTF-8."
            ) from e
        log.info(
            "Uploaded %d records from %s (filtered out %d, malformed lines %d)",
            idx, filename, filtered, bad_json,
        )
        if idx == 0:
            log.warning("%s produced 0 usable records; check its format.", filename)

    if total == 0:
        raise RuntimeError(
            f"No documents uploaded from {data_dir}. "
            "Add at least one of: jasonlemkin_blog.jsonl, jasonlemkinlinkedin.jsonl, "
            "jasonmlemkinyoutubetranscripts.jsonl, saastryoutubetranscripts.jsonl, "
            "jasonlk_originals.jsonl under ./data (mounted in the container)."
        )
    log.info("Uploaded %d total documents to s3://%s/%s/", total, bucket, raw_prefix)


def load_to_chroma():
    import chromadb

    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")
    _check_minio(s3, bucket)

    chunk_keys = [
        k for k in _list_keys(s3, bucket, "chunks/")
        if k.endswith(".json") and not k.startswith("chunks/_")
    ]
    if not chunk_keys:
        raise RuntimeError(
            f"No chunk files under s3://{bucket}/chunks/. "
            "Did trigger_spark_clean succeed and write its output?"
        )
    rows = []
    bad_lines = 0
    for key in chunk_keys:
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        for line_no, line in enumerate(data.strip().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines += 1
                log.warning("Skip bad JSON in s3://%s/%s line %d: %s", bucket, key, line_no, e)
    log.info(
        "Read %d chunk rows from %d file(s) (%d unparseable lines).",
        len(rows), len(chunk_keys), bad_lines,
    )

    # Decide before touching Chroma, so a bad Spark run cannot leave a half-loaded collection.
    _check_embedding_coverage(len(rows), sum(1 for r in rows if not _has_embedding(r)))

    chroma_host = os.environ.get("CHROMA_HOST", "chroma")
    chroma_port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION_NAME", "lemkin_content")
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT

    tenant = os.environ.get("CHROMA_TENANT", DEFAULT_TENANT)
    database = os.environ.get("CHROMA_DATABASE", DEFAULT_DATABASE)
    try:
        client = chromadb.HttpClient(
            host=chroma_host, port=chroma_port, tenant=tenant, database=database
        )
        collection = client.get_or_create_collection(collection_name)
    except Exception as e:  # noqa: BLE001 - re-raised below with a fix hint
        hint = (
            "The Chroma server image and pip chromadb must both be 1.5.5; if the chroma-data "
            "volume is from an older layout, remove it (`docker volume rm <project>_chroma-data`) "
            "and re-run this task. See README 'Chroma default_tenant'."
            if "tenant" in str(e).lower()
            else "Check `docker compose ps chroma` and CHROMA_HOST/CHROMA_PORT."
        )
        raise RuntimeError(
            f"Cannot open Chroma collection '{collection_name}' at {chroma_host}:{chroma_port}: {e}. {hint}"
        ) from e

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
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            except Exception as e:  # noqa: BLE001 - re-raised with the failing batch
                raise RuntimeError(
                    f"Chroma upsert failed for rows {start}-{start + len(chunk) - 1} "
                    f"of {len(rows)} (already upserted: {upserted}): {e}"
                ) from e
            upserted += len(ids)

    log.info(
        "Chroma collection '%s': upserted %d rows with embeddings; skipped %d JSON lines with "
        "missing/empty embedding (out of %d lines read from MinIO).",
        collection_name, upserted, skipped_no_embedding, len(rows),
    )


def mark_training_ready():
    """Promote Spark's training output to a stable filename + write a _READY sentinel.

    Spark writes ``s3://lemkin-processed/training/dataset_jsonl/part-*.txt`` (one
    JSONL line per essay, but a directory + part file because of Spark semantics).
    The host-side fine-tune watcher wants a single, deterministic key
    (``training/dataset.jsonl``) and a sentinel it can poll for change detection.
    """
    s3 = _minio_client()
    bucket = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")
    src_prefix = "training/dataset_jsonl/"
    dest_key = "training/dataset.jsonl"
    sentinel_key = "training/_READY"
    _check_minio(s3, bucket)

    part_keys = [
        k for k in _list_keys(s3, bucket, src_prefix)
        if k.endswith(".txt") and not k.rsplit("/", 1)[-1].startswith("_")
    ]
    if not part_keys:
        raise RuntimeError(
            f"No Spark training output found at s3://{bucket}/{src_prefix}. "
            "Did trigger_spark_clean succeed and write the training dataset?"
        )

    # Spark with coalesce(1) writes a single part file; if there are multiple
    # (e.g. someone bumped the partition count), concatenate in stable order so
    # the watcher always sees one dataset.jsonl regardless of part count.
    part_keys.sort()
    buf = bytearray()
    total_lines = 0
    for k in part_keys:
        body = s3.get_object(Bucket=bucket, Key=k)["Body"].read()
        if not body:
            continue
        if buf and not buf.endswith(b"\n"):
            buf.extend(b"\n")
        buf.extend(body)
        total_lines += body.count(b"\n")
    if not buf.endswith(b"\n"):
        buf.extend(b"\n")

    # Validate before publishing: the sentinel wakes the host-side fine-tune watcher, and a
    # GPU run on an empty / corrupt dataset wastes an hour.
    valid_rows = 0
    bad_rows = 0
    for line in buf.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_rows += 1
            continue
        if isinstance(row, dict) and row.get("instruction") and row.get("output"):
            valid_rows += 1
        else:
            bad_rows += 1
    if valid_rows == 0:
        raise RuntimeError(
            f"Training output under s3://{bucket}/{src_prefix} has no valid "
            "instruction/output rows; not publishing the _READY sentinel."
        )
    if bad_rows:
        log.warning("%d of %d training rows are malformed and will be skipped downstream.",
                    bad_rows, valid_rows + bad_rows)

    s3.put_object(Bucket=bucket, Key=dest_key, Body=bytes(buf))
    sentinel = {
        "ready_at": datetime.utcnow().isoformat() + "Z",
        "dataset_key": dest_key,
        "dataset_bytes": len(buf),
        "approx_rows": total_lines,
        "valid_rows": valid_rows,
        "source_parts": part_keys,
    }
    s3.put_object(
        Bucket=bucket,
        Key=sentinel_key,
        Body=json.dumps(sentinel, indent=2).encode("utf-8"),
    )
    log.info(
        "Promoted %d part file(s) into s3://%s/%s (%d bytes, ~%d rows). Wrote sentinel s3://%s/%s.",
        len(part_keys), bucket, dest_key, len(buf), total_lines, bucket, sentinel_key,
    )


def _ensure_nltk_data() -> None:
    """Make sure the NLTK tokenizer/stopword data is usable, or fail with a clear reason."""
    import nltk

    for pkg, resource in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),  # newer NLTK only; harmless to miss on older
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)  # returns False when offline; verified below
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import sent_tokenize, word_tokenize

        stopwords.words("english")
        sent_tokenize("Ping. Pong.")
        word_tokenize("ping pong")
    except LookupError as e:
        raise RuntimeError(
            "NLTK data (punkt / punkt_tab / stopwords) is missing and could not be downloaded. "
            "The Airflow container needs outbound internet for the first run, or bake the data "
            "into Dockerfile.airflow (python -m nltk.downloader punkt punkt_tab stopwords)."
        ) from e


def extract_persona():
    from collections import Counter

    _ensure_nltk_data()
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    s3 = _minio_client()
    bucket_raw = os.environ.get("MINIO_BUCKET_RAW", "lemkin-raw")
    bucket_processed = os.environ.get("MINIO_BUCKET_PROCESSED", "lemkin-processed")
    _check_minio(s3, bucket_raw, bucket_processed)

    all_text = []
    unreadable = 0
    # Paged listing: a single list_objects_v2 call returns only the first 1000 keys, which
    # silently built the persona from 1000 of ~30k documents (see persona_profile.json meta).
    for key in _list_keys(s3, bucket_raw, "raw/"):
        if not key.endswith(".json"):
            continue
        try:
            rec = json.loads(s3.get_object(Bucket=bucket_raw, Key=key)["Body"].read().decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            unreadable += 1
            log.warning("Skip unreadable raw record s3://%s/%s: %s", bucket_raw, key, e)
            continue
        text = rec.get("text", "") if isinstance(rec, dict) else ""
        if text:
            all_text.append(text)
    if not all_text:
        raise RuntimeError(
            f"No usable documents under s3://{bucket_raw}/raw/. Did extract_to_minio run?"
        )
    log.info("Loaded %d raw documents for persona extraction (%d unreadable).", len(all_text), unreadable)

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
        log.info("Also wrote persona to %s (host ./data when mounted)", local_path)
    except OSError as e:
        log.warning("Could not write local persona file %s: %s (Streamlit will not see it)", local_path, e)

    log.info("Wrote persona_profile.json to MinIO bucket '%s'", bucket_processed)


with DAG(
    dag_id="lemkin_content_pipeline",
    description="Lemkin ETL: local JSONL → MinIO → Spark → Chroma + Persona",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["etl", "lemkin", "persona", "jason-lemkin"],
    # Every task overwrites its output, so re-running after a transient failure (MinIO or
    # Ollama hiccup) is safe. Failures and retries are logged with dag/task/run ids.
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "on_failure_callback": _log_task_failure,
        "on_retry_callback": _log_task_retry,
    },
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
            "MAX_MISSING_EMBEDDING_FRACTION": os.environ.get("MAX_MISSING_EMBEDDING_FRACTION", "0.10"),
            "SFT_CHUNK_OUTPUT_CHARS": os.environ.get("SFT_CHUNK_OUTPUT_CHARS", "1600"),
            "DATASET_MAX_OUTPUT_CHARS": os.environ.get("DATASET_MAX_OUTPUT_CHARS", "2000"),
            "MAX_SEQ_LENGTH": os.environ.get("MAX_SEQ_LENGTH", "512"),
        },
        # Long embed + write; Airflow default is no cap, but this documents intent and avoids surprises if a cap is set globally.
        execution_timeout=timedelta(hours=12),
        verbose=True,
        # One retry only: a full embed pass can take hours, and the Spark job now fails fast
        # (Ollama preflight) when the cause is a down or misconfigured Ollama.
        retries=1,
        retry_delay=timedelta(minutes=5),
    )

    task_chroma = PythonOperator(
        task_id="load_to_chroma",
        python_callable=load_to_chroma,
    )

    task_persona = PythonOperator(
        task_id="extract_persona",
        python_callable=extract_persona,
    )

    task_training_ready = PythonOperator(
        task_id="mark_training_ready",
        python_callable=mark_training_ready,
    )

    task_extract >> task_spark >> [task_chroma, task_training_ready]
    task_extract >> task_persona
