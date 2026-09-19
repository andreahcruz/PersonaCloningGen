import io
import json
import logging

import pytest

import lemkin_pipeline_dag as dag


class FakeS3:
    """Just enough of the boto3 S3 client for the DAG helpers, with 2-key pages to exercise paging."""

    def __init__(self, objects=None, missing_buckets=()):
        self.objects = dict(objects or {})
        self.puts = {}
        self.missing_buckets = set(missing_buckets)

    def head_bucket(self, Bucket):
        if Bucket in self.missing_buckets:
            raise RuntimeError(f"NoSuchBucket: {Bucket}")

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        outer = self

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in outer.objects if k.startswith(Prefix))
                for i in range(0, len(keys), 2):
                    yield {"Contents": [{"Key": k} for k in keys[i:i + 2]]}
                if not keys:
                    yield {}

        return Paginator()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.puts[Key] = Body
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()


def jl(*rows):
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


# ── helpers ───────────────────────────────────────────────────────────────
def test_require_env_names_every_missing_variable(monkeypatch):
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.setenv("MINIO_ACCESS_KEY", "k")
    monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        dag._require_env("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")
    assert "MINIO_ENDPOINT" in str(exc.value) and "MINIO_SECRET_KEY" in str(exc.value)
    assert "MINIO_ACCESS_KEY" not in str(exc.value)


def test_check_minio_reports_the_bucket_and_a_fix():
    with pytest.raises(RuntimeError) as exc:
        dag._check_minio(FakeS3(missing_buckets={"lemkin-raw"}), "lemkin-processed", "lemkin-raw")
    assert "lemkin-raw" in str(exc.value) and "minio-setup" in str(exc.value)


def test_list_keys_pages_past_the_first_response():
    s3 = FakeS3({f"raw/doc_{i:03d}.json": b"{}" for i in range(7)})
    keys = dag._list_keys(s3, "b", "raw/")
    assert len(keys) == 7, "must return every key, not just the first page"
    assert dag._list_keys(s3, "b", "nothing/") == []


@pytest.mark.parametrize(
    "total,missing,ok",
    [(100, 0, True), (100, 10, True), (100, 11, False), (10, 10, False)],
)
def test_embedding_coverage_gate(total, missing, ok, monkeypatch):
    monkeypatch.setattr(dag, "MAX_MISSING_EMBEDDING_FRACTION", 0.10)
    if ok:
        dag._check_embedding_coverage(total, missing)
    else:
        with pytest.raises(RuntimeError, match="no embedding"):
            dag._check_embedding_coverage(total, missing)


def test_embedding_coverage_gate_rejects_an_empty_run():
    with pytest.raises(RuntimeError, match="No chunk rows"):
        dag._check_embedding_coverage(0, 0)


def test_failure_callback_logs_ids_and_survives_a_sparse_context(caplog):
    caplog.set_level(logging.ERROR, logger="airflow.task")
    ti = type("TI", (), {"dag_id": "lemkin_content_pipeline", "task_id": "load_to_chroma",
                         "try_number": 2, "log_url": "http://airflow/log"})()
    dag._log_task_failure({"task_instance": ti, "run_id": "manual__1", "exception": ValueError("boom")})
    dag._log_task_failure({})  # must not raise inside a callback
    text = caplog.text
    assert "load_to_chroma" in text and "manual__1" in text and "boom" in text


# ── extract_to_minio ──────────────────────────────────────────────────────
@pytest.fixture
def extract_env(monkeypatch, tmp_path):
    s3 = FakeS3()
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    monkeypatch.setenv("LEMKIN_DATA_DIR", str(tmp_path))
    return s3, tmp_path


def test_extract_skips_bad_lines_but_counts_and_logs_them(extract_env, caplog):
    s3, data_dir = extract_env
    caplog.set_level(logging.INFO, logger="airflow.task")
    (data_dir / "jasonlemkin_blog.jsonl").write_bytes(
        b'{"title": "T", "content": "hello world", "date": "2020"}\n'
        b"this is not json\n"
        b"[1, 2, 3]\n"
        b'{"title": "empty", "content": ""}\n'
    )
    dag.extract_to_minio()
    assert list(s3.puts) == ["raw/jasonlemkin_blog_000000.json"]
    assert "Skip bad JSON in jasonlemkin_blog.jsonl line 2" in caplog.text
    assert "non-object JSON" in caplog.text
    assert "Uploaded 1 records from jasonlemkin_blog.jsonl (filtered out 1, malformed lines 2)" in caplog.text


def test_extract_fails_clearly_when_no_input_exists(extract_env):
    with pytest.raises(RuntimeError, match="No documents uploaded"):
        dag.extract_to_minio()


def test_extract_reports_a_file_that_is_not_utf8(extract_env):
    _, data_dir = extract_env
    (data_dir / "jasonlemkin_blog.jsonl").write_bytes(b'{"content": "ok"}\n\xff\xfe broken\n')
    with pytest.raises(RuntimeError, match="not valid UTF-8"):
        dag.extract_to_minio()


def test_extract_stops_before_reading_files_if_minio_is_down(monkeypatch, tmp_path):
    monkeypatch.setattr(dag, "_minio_client", lambda: FakeS3(missing_buckets={"lemkin-raw"}))
    monkeypatch.setenv("LEMKIN_DATA_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="lemkin-raw"):
        dag.extract_to_minio()


# ── load_to_chroma ────────────────────────────────────────────────────────
class FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kw):
        self.upserts.append(kw)


def _chunk(i, emb=True):
    row = {"essay_id": i, "chunk_index": 0, "source": "blog", "chunk_text": f"t{i}", "title": "T"}
    if emb:
        row["embedding"] = [0.1, 0.2]
    return row


def _patch_chroma(monkeypatch, collection=None, forbid=False):
    import chromadb

    def client(**kw):
        if forbid:
            pytest.fail("Chroma must not be touched when the embedding gate fails")
        return type("C", (), {"get_or_create_collection": lambda self, n: collection})()

    monkeypatch.setattr(chromadb, "HttpClient", client)


def test_load_refuses_to_touch_chroma_when_too_many_embeddings_are_missing(monkeypatch):
    rows = [_chunk(0), _chunk(1, emb=False), _chunk(2, emb=False), _chunk(3, emb=False)]
    s3 = FakeS3({"chunks/part-0.json": jl(*rows)})
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    _patch_chroma(monkeypatch, forbid=True)
    with pytest.raises(RuntimeError, match="no embedding"):
        dag.load_to_chroma()


def test_load_upserts_good_rows_and_ignores_success_marker_and_bad_lines(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="airflow.task")
    good = jl(_chunk(0), _chunk(1), _chunk(2, emb=False)) + b"{broken json\n"
    s3 = FakeS3({"chunks/part-0.json": good, "chunks/_SUCCESS": b""})
    collection = FakeCollection()
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    monkeypatch.setattr(dag, "MAX_MISSING_EMBEDDING_FRACTION", 0.5)
    _patch_chroma(monkeypatch, collection)

    dag.load_to_chroma()

    ids = [i for call in collection.upserts for i in call["ids"]]
    assert ids == ["blog_0_0", "blog_1_0"]
    assert "1 unparseable lines" in caplog.text
    assert "upserted 2 rows" in caplog.text


def test_load_explains_a_chroma_tenant_error(monkeypatch):
    s3 = FakeS3({"chunks/part-0.json": jl(_chunk(0))})
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    import chromadb

    def boom(**kw):
        raise ValueError("Could not connect to tenant default_tenant")

    monkeypatch.setattr(chromadb, "HttpClient", boom)
    with pytest.raises(RuntimeError, match="chroma-data"):
        dag.load_to_chroma()


def test_load_fails_when_spark_wrote_nothing(monkeypatch):
    monkeypatch.setattr(dag, "_minio_client", lambda: FakeS3())
    with pytest.raises(RuntimeError, match="trigger_spark_clean"):
        dag.load_to_chroma()


def test_load_reports_the_failing_batch(monkeypatch):
    s3 = FakeS3({"chunks/part-0.json": jl(_chunk(0), _chunk(1))})

    class Failing:
        def upsert(self, **kw):
            raise ValueError("disk full")

    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    _patch_chroma(monkeypatch, Failing())
    with pytest.raises(RuntimeError, match="upsert failed for rows 0-1"):
        dag.load_to_chroma()


# ── mark_training_ready ───────────────────────────────────────────────────
TRAIN = "training/dataset_jsonl/part-00000.txt"


def test_training_ready_publishes_dataset_and_sentinel(monkeypatch):
    rows = jl({"instruction": "Write ...", "input": "", "output": "body one"},
              {"instruction": "Write ...", "input": "", "output": "body two"})
    s3 = FakeS3({TRAIN: rows})
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    dag.mark_training_ready()
    assert "training/dataset.jsonl" in s3.puts
    sentinel = json.loads(s3.puts["training/_READY"])
    assert sentinel["valid_rows"] == 2 and sentinel["approx_rows"] == 2


def test_training_ready_never_wakes_the_watcher_for_a_bad_dataset(monkeypatch):
    s3 = FakeS3({TRAIN: b"not json\n" + jl({"instruction": "x"})})  # second row has no output
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    with pytest.raises(RuntimeError, match="no valid"):
        dag.mark_training_ready()
    assert "training/_READY" not in s3.puts and "training/dataset.jsonl" not in s3.puts


def test_training_ready_warns_about_partially_bad_rows(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="airflow.task")
    good = {"instruction": "Write ...", "output": "body"}
    s3 = FakeS3({TRAIN: jl(good) + b"garbage\n"})
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    dag.mark_training_ready()
    assert "1 of 2 training rows are malformed" in caplog.text


def test_training_ready_does_not_split_rows_on_unicode_line_separators(monkeypatch, caplog):
    # Found in the first real run: 6 of 19,961 rows contain U+2028, which str.splitlines() treats
    # as a line break, producing false "malformed row" warnings.
    caplog.set_level(logging.WARNING, logger="airflow.task")
    rows = [
        {"instruction": "Write ...", "input": "", "output": "first line second line"},
        {"instruction": "Write ...", "input": "", "output": "plain"},
    ]
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    assert " ".encode("utf-8") in body
    s3 = FakeS3({TRAIN: body})
    monkeypatch.setattr(dag, "_minio_client", lambda: s3)
    dag.mark_training_ready()
    assert json.loads(s3.puts["training/_READY"])["valid_rows"] == 2
    assert "malformed" not in caplog.text


def test_training_ready_still_fails_when_spark_wrote_nothing(monkeypatch):
    monkeypatch.setattr(dag, "_minio_client", lambda: FakeS3())
    with pytest.raises(RuntimeError, match="No Spark training output"):
        dag.mark_training_ready()


# ── extract_persona ───────────────────────────────────────────────────────
def test_persona_fails_clearly_without_raw_documents(monkeypatch):
    pytest.importorskip("nltk")
    monkeypatch.setattr(dag, "_ensure_nltk_data", lambda: None)
    monkeypatch.setattr(dag, "_minio_client", lambda: FakeS3())
    with pytest.raises(RuntimeError, match="extract_to_minio"):
        dag.extract_persona()


def test_nltk_data_failure_has_an_actionable_message(monkeypatch):
    # Offline container: no data on disk (find fails) and the download returns False.
    nltk = pytest.importorskip("nltk")
    monkeypatch.setattr(nltk.data, "find", lambda r: (_ for _ in ()).throw(LookupError(r)))
    monkeypatch.setattr(nltk, "download", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="NLTK data"):
        dag._ensure_nltk_data()
