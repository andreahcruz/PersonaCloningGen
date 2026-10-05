"""The frozen balanced file is the keep-breaks replay with Windows CRLF."""
import pytest

from host_finetune.diagnose_balance_parity import EXPECTED_SHA256, diagnose


@pytest.fixture(scope="module")
def report():
    return diagnose()


def test_membership_and_order_match(report):
    assert report["frozen_rows"] == 31328
    assert report["replay_rows"] == 31328
    assert report["intersection"] == 31328
    assert report["frozen_only"] == 0
    assert report["replay_only"] == 0
    assert report["membership_identical"] is True
    assert report["order_identical"] is True
    assert report["parsed_identical"] is True
    assert report["value_diffs_in_paired_order"] == 0
    assert report["key_order_diffs_in_paired_order"] == 0


def test_crlf_serializer_matches_frozen_bytes(report):
    by_name = {item["name"]: item for item in report["candidates"]}
    crlf = by_name["ensure_ascii_false_crlf"]
    lf = by_name["ensure_ascii_false_lf"]
    copied = by_name["parent_line_bytes_plus_crlf"]
    assert crlf["sha256"] == EXPECTED_SHA256
    assert crlf["matches_frozen_bytes"] is True
    assert copied["matches_frozen_bytes"] is True
    assert lf["sha256"] == "60481f38fee42dbfa6c980e65cb7d6ce620e8f9a920f422701487f1922f7486a"
    assert lf["matches_frozen_hash"] is False
    assert report["frozen_newlines"]["crlf"] == 31328
    assert report["frozen_newlines"]["bom"] is False
    assert report["level"] == 1
    assert report["exact_hash_reproduced"] is True
