"""Offline owning workflow checks using the actual DPP source interface."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from data_pipeline_pretrain.pipeline import root_data as rd
from datatrove.executor.local import LocalPipelineExecutor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from tokenization_scripts.preprocess_megatron import TokenizationJob, get_args, run_job

ROOT = rd.ArtifactRef.parse("data/offline-text@v1")
OUTPUT = rd.ArtifactRef.parse("data/offline-tokens@v1")


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


def prepare(
    tmp_path: Path, texts: list[str], *, wide: bool = False, text_column: str = "text"
) -> TokenizationJob:
    assets = tmp_path / "assets"
    assets.mkdir()
    git(assets, "init", "--quiet")
    git(assets, "remote", "add", "origin", "https://example.invalid/offline-fixture")
    vocabulary = {"[UNK]": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4}
    if wide:
        # A dense vocabulary crosses the uint16 boundary without model downloads.
        vocabulary.update({f"token{i}": i for i in range(5, 65537)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    tokenizer.enable_truncation(max_length=3)
    tokenizer.enable_padding(length=8)
    asset = assets / "tokenizer.json"
    tokenizer.save(str(asset))
    git(assets, "add", "tokenizer.json")
    git(
        assets,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Pin offline tokenizer",
    )
    revision = git(assets, "rev-parse", "HEAD")
    root = tmp_path / "collections"
    writer = rd.InitialRootParquetWriter(
        root,
        ROOT,
        schema=pa.schema([(text_column, pa.string())]),
        tasks=1,
        text_key=text_column,
    )
    writer.prepare()
    list(writer.run({text_column: text} for text in texts))
    executor = LocalPipelineExecutor(
        [], tasks=1, logging_dir=str(tmp_path / "root-logs")
    )
    executor.mark_rank_as_completed(0)
    writer.finalize(executor)
    return TokenizationJob(
        root,
        ROOT,
        ROOT,
        OUTPUT,
        asset,
        revision,
        tmp_path / "logs",
        prefix_tokens=(1,),
        suffix_tokens=(2,),
        tasks=4,
        workers=2,
        batch_size=2,
        batch_bytes=3,
        max_sequences_per_shard=1,
    )


@pytest.mark.parametrize("wide", [False, True])
def test_offline_native_fresh_resume_and_corruption(tmp_path: Path, wide: bool) -> None:
    job = prepare(tmp_path, ["a b a", "b", "a b"], wide=wide)
    tokens, lengths = run_job(job)
    collection = rd.TokenCollection(job.data_root, ROOT, OUTPUT)
    collection.validate()
    assert [row.tolist() for row in collection.sequences()] == [
        [1, 3, 4, 3, 2],
        [1, 4, 2],
        [1, 3, 4, 2],
    ]
    assert all(
        row.dtype == np.dtype("<i4" if wide else "<u2")
        for row in collection.sequences()
    )
    assert list(
        rd.ParquetCollection(job.data_root, ROOT, rd.token_length_ref(OUTPUT)).rows(
            [rd.LENGTH_FIELD]
        )
    ) == [{rd.LENGTH_FIELD: length} for length in (5, 3, 4)]
    assert tokens.semantics["token_count"] == 12
    assert tokens.semantics["encoding"]["tokenizer"]["commit"] == job.tokenizer_revision
    assert tokens.complete and lengths.complete
    assert not list(job.data_root.rglob("*.map"))
    assert os.environ["RAYON_NUM_THREADS"] == "1"
    before = {p: p.read_bytes() for p in job.data_root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="explicit resume"):
        run_job(job)
    run_job(replace(job, resume=True))
    assert before == {
        p: p.read_bytes() for p in job.data_root.rglob("*") if p.is_file()
    }
    binary = next(
        p
        for p in rd.collection_path(job.data_root, ROOT, OUTPUT).rglob("*.bin")
        if p.stat().st_size
    )
    binary.write_bytes(b"x")
    with pytest.raises(ValueError):
        run_job(replace(job, resume=True))
    with pytest.raises(rd.IncompleteCollectionError):
        rd.read_metadata(job.data_root, ROOT, OUTPUT)


def test_changed_asset_fails_before_outputs(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["a"])
    job.tokenizer_file.write_text("modified local asset")
    with pytest.raises(ValueError, match="bytes differ"):
        run_job(job)
    assert not rd.collection_path(job.data_root, ROOT, OUTPUT).exists()


def test_empty_body_propagates_and_outputs_stay_incomplete(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["   "])
    with pytest.raises(ValueError, match="empty body"):
        run_job(replace(job, tasks=1, workers=1))
    for ref in (OUTPUT, rd.token_length_ref(OUTPUT)):
        with pytest.raises(rd.IncompleteCollectionError):
            rd.read_metadata(job.data_root, ROOT, ref)


def test_wrong_root_and_missing_asset_fail_without_token_output(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["a"])
    with pytest.raises(FileNotFoundError):
        run_job(replace(job, tokenizer_file=tmp_path / "missing.json"))
    with pytest.raises((ValueError, FileNotFoundError)):
        run_job(replace(job, root_ref=rd.ArtifactRef.parse("data/missing-root@v1")))
    assert not rd.collection_path(job.data_root, ROOT, OUTPUT).exists()


def test_cli_help_defaults_and_invalid_resources(tmp_path: Path) -> None:
    help_text = subprocess.check_output(
        [sys.executable, "-m", "tokenization_scripts.preprocess_megatron", "--help"],
        text=True,
    )
    assert "UTF-8 bytes" in help_text and "Rayon threads" in help_text
    arguments = [
        "--data-root",
        str(tmp_path),
        "--root-ref",
        str(ROOT),
        "--input-ref",
        str(ROOT),
        "--output-ref",
        str(OUTPUT),
        "--tokenizer-file",
        "tokenizer.json",
        "--tokenizer-revision",
        "a" * 40,
        "--logging-dir",
        "logs",
        "--prefix-tokens",
        "1",
        "--suffix-tokens",
        "2",
    ]
    job = get_args(arguments)
    assert (job.tasks, job.workers, job.threads, job.batch_size, job.batch_bytes) == (
        1,
        1,
        1,
        1000,
        33554432,
    )
    with pytest.raises(SystemExit):
        get_args([*arguments, "--workers", "2"])
    with pytest.raises(SystemExit):
        get_args([*arguments, "--batch-bytes", "0"])


def test_actual_cli_uses_declared_primary_and_explicit_boundaries(
    tmp_path: Path,
) -> None:
    job = prepare(tmp_path, ["a b"], text_column="body")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tokenization_scripts.preprocess_megatron",
            "--data-root",
            str(job.data_root),
            "--root-ref",
            str(job.root_ref),
            "--input-ref",
            str(job.input_ref),
            "--output-ref",
            str(job.output_ref),
            "--tokenizer-file",
            str(job.tokenizer_file),
            "--tokenizer-revision",
            job.tokenizer_revision,
            "--logging-dir",
            str(job.logging_dir),
            "--prefix-tokens",
            "",
            "--suffix-tokens",
            "2",
            "--tasks",
            "2",
            "--workers",
            "1",
            "--threads",
            "2",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "1 rows, 3 stored tokens" in result.stdout
    collection = rd.TokenCollection(job.data_root, job.root_ref, job.output_ref)
    collection.validate()
    assert [sequence.tolist() for sequence in collection.sequences()] == [[3, 4, 2]]
    assert collection.metadata.semantics["prefix_tokens"] == []
