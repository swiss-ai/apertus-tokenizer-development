"""A retry must preserve the full dump population and its exact pins."""

import hashlib
import json

import pytest

from tokenization_scripts.hplt_finite_score_resume import check


def fixture(tmp_path):
    path = tmp_path / "dumps/jpn_Jpan/paths_file_0.txt"
    path.parent.mkdir(parents=True)
    content = b"examples/jpn_Jpan/000001.parquet\n"
    path.write_bytes(content)
    pins = {"config": "fixed", "implementation_commit": "code"}
    row = {
        "pins": pins,
        "dumps": {"jpn_Jpan/paths_file_0.txt": hashlib.sha256(content).hexdigest()},
    }
    (tmp_path / "HPLT_DUMPS.json").write_text(json.dumps(row))
    return path, pins


def test_pending_to_completed_is_the_same_population(tmp_path):
    path, pins = fixture(tmp_path)
    check(tmp_path, pins)
    completed = tmp_path / "completed-dumps/jpn_Jpan/paths_file_0.txt"
    completed.parent.mkdir(parents=True)
    path.rename(completed)
    check(tmp_path, pins)


def test_missing_inventory_is_not_completion(tmp_path):
    path, pins = fixture(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match="missing"):
        check(tmp_path, pins)


def test_dump_and_producer_changes_fail(tmp_path):
    path, pins = fixture(tmp_path)
    with pytest.raises(ValueError, match="pins changed"):
        check(tmp_path, {**pins, "implementation_commit": "different"})
    path.write_text("a different population\n")
    with pytest.raises(ValueError, match="inventory changed"):
        check(tmp_path, pins)


def test_double_completion_fails(tmp_path):
    path, pins = fixture(tmp_path)
    duplicate = tmp_path / "completed-dumps/jpn_Jpan/paths_file_0.txt"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        check(tmp_path, pins)
