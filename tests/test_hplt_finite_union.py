"""Keep old token bytes, replay new provenance, and reject incomplete unions."""

from array import array

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data_pipeline_pretrain.pipeline.tokens.megatron_index import pack_megatron_index
from data_pipeline_pretrain.pipeline.tokens.megatron_tokenizer import (
    megatron_post_processor,
)
from data_pipeline_pretrain.pipeline.tokens.token_map import pack_token_map
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from tokenization_scripts import hplt_finite_score_union as union


def pair(prefix, documents):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray([token for doc in documents for token in doc], dtype="<u4")
    prefix.with_suffix(".bin").write_bytes(values.tobytes())
    prefix.with_suffix(".idx").write_bytes(
        pack_megatron_index([len(doc) for doc in documents], 4)
    )


def recovered_fixture(root, monkeypatch):
    tokenizer_path = root / "tokenizer.json"
    tokenizer = Tokenizer(
        WordLevel(
            {"<UNK>": 0, "<BOS>": 1, "<EOS>": 2, "alpha": 3, "beta": 4},
            unk_token="<UNK>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tokenizer_path))
    tokenizer.post_processor = megatron_post_processor()
    monkeypatch.setattr(union, "TOKENIZER_SHA256", union.sha(tokenizer_path))
    prepared = root / "prepared"
    source = prepared / "examples/jpn_Jpan/source.parquet"
    source.parent.mkdir(parents=True)
    texts = ["alpha beta", "beta"]
    pq.write_table(pa.table({"id": ["one", "two"], "text": texts}), source)
    prefix = root / "recovered/jpn_Jpan/dump-0/tokens"
    documents = [tokenizer.encode(text).ids for text in texts]
    pair(prefix, documents)
    file = {
        "path": source.relative_to(prepared).as_posix(),
        "bytes": source.stat().st_size,
        "num_rows": 2,
        "num_row_groups": 1,
        "footer_sha256": "unused-in-this-unit-test",
        "first_sequence": 0,
        "emitted_sequences": 2,
        "emitted_rows": 2,
    }
    manifest = {
        "raw_dataset_root": str(prepared),
        "tokenizer": {"sha256": union.TOKENIZER_SHA256},
        "index_sha256": union.sha(prefix.with_suffix(".idx")),
    }
    prefix.with_suffix(".map").write_bytes(
        pack_token_map([file], array("Q", [0, 1]), sum(map(len, documents)), manifest)
    )
    return prefix, tokenizer_path


def test_preserve_old_bytes_and_replay_new_source_before_sealing(tmp_path, monkeypatch):
    new, tokenizer = recovered_fixture(tmp_path, monkeypatch)
    old = tmp_path / "old/fin_Latn/tokens"
    pair(old, [[1, 3, 2]])
    reserved = tmp_path / "old/_reserved/long-context/tokens"
    pair(reserved, [[1, 3, 4, 2]])
    assert union.discover(tmp_path / "old") == ["fin_Latn/tokens"]
    old_report = union.qualify(
        (
            str(tmp_path / "old"),
            "fin_Latn/tokens",
            "original",
            str(tmp_path / "control"),
            str(tokenizer),
        )
    )
    new_report = union.qualify(
        (
            str(tmp_path / "recovered"),
            "jpn_Jpan/dump-0/tokens",
            "recovered",
            str(tmp_path / "control"),
            str(tokenizer),
        )
    )
    assert [row["id"] for row in new_report["examples"]] == ["one", "two"]
    assert all(row["exact_reencoding"] for row in new_report["examples"])
    output = tmp_path / "union"
    counts = {
        "combined": {"source_tokens": 14, "reserved_tokens": 4, "kept_tokens": 10}
    }
    result = union.seal_union(
        output,
        [old_report, new_report],
        counts,
        {"tokenizer_sha256": union.TOKENIZER_SHA256},
        "dclm-10",
    )
    assert result["complete"]
    assert (output / "original/fin_Latn/tokens.bin").is_symlink()
    assert (output / "original/fin_Latn/tokens.bin").read_bytes() == old.with_suffix(
        ".bin"
    ).read_bytes()
    assert (
        output / "recovered/jpn_Jpan/dump-0/tokens.map"
    ).read_bytes() == new.with_suffix(".map").read_bytes()
    assert not (output / "_reserved").exists()
    prior = (output / "_SUCCESS.json").read_bytes()
    union.seal_union(
        output,
        [old_report, new_report],
        counts,
        {"tokenizer_sha256": union.TOKENIZER_SHA256},
        "dclm-10",
    )
    assert (output / "_SUCCESS.json").read_bytes() == prior
    old.with_suffix(".bin").write_bytes(np.asarray([1, 4, 2], dtype="<u4").tobytes())
    with pytest.raises(ValueError, match="parent changed"):
        union.seal_union(
            output,
            [old_report, new_report],
            counts,
            {"tokenizer_sha256": union.TOKENIZER_SHA256},
            "dclm-10",
        )


def test_recovered_token_corruption_fails_even_when_counts_match(tmp_path, monkeypatch):
    prefix, tokenizer = recovered_fixture(tmp_path, monkeypatch)
    values = np.frombuffer(prefix.with_suffix(".bin").read_bytes(), dtype="<u4").copy()
    values[1] = 4
    prefix.with_suffix(".bin").write_bytes(values.tobytes())
    with pytest.raises(ValueError, match="differ from source encoding"):
        union.qualify(
            (
                str(tmp_path / "recovered"),
                "jpn_Jpan/dump-0/tokens",
                "recovered",
                str(tmp_path / "control"),
                str(tokenizer),
            )
        )
    assert not list((tmp_path / "control").glob("*.json"))


def test_missing_pair_and_ledger_drift_fail(tmp_path, monkeypatch):
    root = tmp_path / "old"
    prefix = root / "tokens"
    pair(prefix, [[1, 3, 2]])
    prefix.with_suffix(".bin").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        union.discover(root)
    prefix.with_suffix(".bin").write_bytes(np.asarray([1, 3, 2], dtype="<u4").tobytes())
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("fixture")
    monkeypatch.setattr(union, "TOKENIZER_SHA256", union.sha(tokenizer))
    monkeypatch.setattr(
        "sys.argv",
        [
            "union",
            "original",
            "--variant",
            "dclm-10",
            "--original",
            str(root),
            "--tokenizer",
            str(tokenizer),
            "--control",
            str(tmp_path / "control"),
            "--workers",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="accepted ledger"):
        union.main()
    assert not (tmp_path / "control/original.json").exists()
