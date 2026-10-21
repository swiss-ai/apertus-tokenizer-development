"""Offline owning workflow checks using the actual DPP source interface."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from data_pipeline_pretrain.pipeline import root_data as rd
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from tokenization_scripts.tokenize import (
    EncodeTokens,
    TokenizationJob,
    get_args,
    load_asset,
    run_job,
)

ROOT = "data/offline-text@v1"
OUTPUT = "data/offline-tokens@v1"


def git(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


class TextRows(PipelineStep):
    """Small source fixture executed through the native pipeline lifecycle."""

    def __init__(self, texts: list[str], column: str) -> None:
        super().__init__()
        self.texts, self.column = texts, column

    def run(self, data=None, rank=0, world_size=1):
        for index, text in enumerate(self.texts):
            yield Document(text=text, id=str(index), metadata={self.column: text})


class AlignedText(PipelineStep):
    """Give an aligned text fixture contents distinguishable from its root."""

    def run(self, data, rank=0, world_size=1):
        for document in data:
            document.text = "b"
            yield document


def token_spec(collection: rd.ParquetCollection) -> rd.TokenSpec:
    metadata = collection.schema.field("tokens").metadata
    assert metadata is not None
    return rd.TokenSpec.from_dict(json.loads(metadata[b"dpp:token_spec"]))


def sequences(job: TokenizationJob) -> list[rd.TokenArray]:
    collection = rd.ParquetCollection(job.data_store, job.root_ref, job.output_ref)
    return list(
        collection.sequences_at(
            range(collection.row_count), spec=token_spec(collection)
        )
    )


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
    writer = rd.RootWriter(
        root,
        ROOT,
        schema=pa.schema([(text_column, pa.string())]),
    )
    executor = LocalPipelineExecutor(
        [TextRows(texts, text_column), writer],
        tasks=1,
        workers=1,
        logging_dir=str(tmp_path / "root-logs"),
    )
    writer.prepare(executor)
    executor.run()
    writer.finalize(executor)
    return TokenizationJob(
        data_store=root,
        root_ref=ROOT,
        output_ref=OUTPUT,
        tokenizer_file=asset,
        tokenizer_revision=revision,
        logging_dir=tmp_path / "logs",
        prefix_tokens=(1,),
        suffix_tokens=(2,),
        text_column=text_column,
        tasks=4,
        workers=2,
        batch_size=2,
        batch_bytes=3,
    )


@pytest.mark.parametrize("wide", [False, True])
def test_parquet_fresh_resume_and_corruption(
    tmp_path: Path, wide: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = prepare(tmp_path, ["a b a", "token65536" if wide else "b", "a b"], wide=wide)
    finalize = rd.RootWriter.finalize

    def interrupt_finalization(
        writer: rd.RootWriter, executor: LocalPipelineExecutor
    ) -> None:
        raise RuntimeError("interrupted before finalization")

    monkeypatch.setattr(rd.RootWriter, "finalize", interrupt_finalization)
    with pytest.raises(RuntimeError, match="interrupted before finalization"):
        run_job(job)
    assert (
        rd.read_metadata(job.data_store, ROOT, OUTPUT, require_complete=False).result
        is None
    )
    payload = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in job.data_store.rglob("*")
        if p.suffix == ".parquet"
    }
    with pytest.raises(
        (FileExistsError, ValueError), match="occupied output|native completions"
    ):
        run_job(job)
    monkeypatch.setattr(rd.RootWriter, "finalize", finalize)
    tokens = run_job(replace(job, input_ref=ROOT, resume=True))
    assert payload == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in payload}
    collection = rd.ParquetCollection(job.data_store, ROOT, OUTPUT)
    assert [row.tolist() for row in sequences(job)] == [
        [1, 3, 4, 3, 2],
        [1, 65536 if wide else 4, 2],
        [1, 3, 4, 2],
    ]
    assert all(row.dtype == np.dtype("<i4") for row in sequences(job))
    assert [
        row
        for batch in rd.ParquetCollection(job.data_store, ROOT, OUTPUT).batches(
            ["stored_tokens"]
        )
        for row in batch.to_pylist()
    ] == [{"stored_tokens": length} for length in (5, 3, 4)]
    assert (
        token_spec(collection).encoding["tokenizer"]["commit"] == job.tokenizer_revision
    )
    assert tokens.result == {"row_count": 3}
    token_path = rd.collection_path(job.data_store, ROOT, OUTPUT)
    assert len(list(token_path.glob("*.parquet"))) == job.tasks
    assert len(list(token_path.rglob("_metadata.json"))) == 1
    joined = rd.RootParquetReader(job.data_store, ROOT, {OUTPUT: ["stored_tokens"]})
    assert [row.metadata["stored_tokens"] for row in joined.run()] == [5, 3, 4]
    assert not list(job.data_store.rglob("*.map"))
    assert os.environ["RAYON_NUM_THREADS"] == "1"
    before = {p: p.read_bytes() for p in job.data_store.rglob("*") if p.is_file()}
    with pytest.raises(
        (FileExistsError, ValueError), match="occupied output|native completions"
    ):
        run_job(job)
    with pytest.raises(FileExistsError, match="completed output is immutable"):
        run_job(replace(job, resume=True))
    assert before == {
        p: p.read_bytes() for p in job.data_store.rglob("*") if p.is_file()
    }
    parquet = max(token_path.glob("*.parquet"), key=lambda path: path.stat().st_size)
    parquet.write_bytes(b"x")
    with pytest.raises((ValueError, OSError)):
        sequences(job)
    assert rd.read_metadata(job.data_store, ROOT, OUTPUT) == tokens


def test_changed_asset_fails_before_outputs(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["a"])
    job.tokenizer_file.write_bytes(job.tokenizer_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes differ"):
        run_job(job)
    assert not rd.collection_path(job.data_store, ROOT, OUTPUT).exists()


def commit_asset(asset: Path) -> str:
    git(asset.parent, "add", asset.name)
    git(
        asset.parent,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Pin updated tokenizer asset",
    )
    return git(asset.parent, "rev-parse", "HEAD")


def lfs_pointer(payload: bytes) -> bytes:
    return (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
        f"size {len(payload)}\n"
    ).encode("ascii")


def pin_lfs(job: TokenizationJob) -> tuple[TokenizationJob, bytes, bytes]:
    payload = job.tokenizer_file.read_bytes()
    pointer = lfs_pointer(payload)
    job.tokenizer_file.write_bytes(pointer)
    revision = commit_asset(job.tokenizer_file)
    job.tokenizer_file.write_bytes(payload)
    return replace(job, tokenizer_revision=revision), payload, pointer


def test_materialized_lfs_asset_runs_with_exact_boundaries_and_lengths(
    tmp_path: Path,
) -> None:
    job, payload, _ = pin_lfs(prepare(tmp_path, ["a b", "b"]))
    asset, identity = load_asset(job.tokenizer_file, job.tokenizer_revision)
    assert asset == payload
    assert identity == {
        "repository": "https://example.invalid/offline-fixture",
        "commit": job.tokenizer_revision,
        "path": "tokenizer.json",
    }
    tokens = run_job(replace(job, tasks=1, workers=1))
    assert tokens.result is not None
    assert [row.tolist() for row in sequences(job)] == [[1, 3, 4, 2], [1, 4, 2]]
    assert [
        row
        for batch in rd.ParquetCollection(job.data_store, ROOT, OUTPUT).batches(
            ["stored_tokens"]
        )
        for row in batch.to_pylist()
    ] == [{"stored_tokens": 4}, {"stored_tokens": 3}]


@pytest.mark.parametrize(
    "fault, message",
    [
        ("pointer", "unmaterialized.*git.*lfs.*pull"),
        ("missing", "missing.*git lfs pull"),
        ("corruption", "SHA-256 differs"),
        ("wrong-size", "size differs"),
        ("wrong-oid", "SHA-256 differs"),
        ("declared-size", "size differs"),
        ("invalid-pointer", "invalid Git LFS pointer"),
    ],
)
def test_lfs_asset_failures_precede_output_mutation(
    tmp_path: Path, fault: str, message: str
) -> None:
    job, payload, pointer = pin_lfs(prepare(tmp_path, ["a"]))
    if fault == "pointer":
        job.tokenizer_file.write_bytes(pointer)
    elif fault == "missing":
        job.tokenizer_file.unlink()
    elif fault == "corruption":
        job.tokenizer_file.write_bytes(b"[" + payload[1:])
    elif fault == "wrong-size":
        job.tokenizer_file.write_bytes(payload + b" ")
    else:
        if fault == "wrong-oid":
            pointer = pointer.replace(
                hashlib.sha256(payload).hexdigest().encode(), b"0" * 64
            )
        elif fault == "declared-size":
            pointer = pointer.replace(
                f"size {len(payload)}".encode(), f"size {len(payload) + 1}".encode()
            )
        else:
            pointer = pointer.replace(b"oid sha256:", b"oid sha512:")
        job.tokenizer_file.write_bytes(pointer)
        job = replace(job, tokenizer_revision=commit_asset(job.tokenizer_file))
        job.tokenizer_file.write_bytes(payload)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run_job(job)
    assert not rd.collection_path(job.data_store, ROOT, OUTPUT).exists()


@pytest.mark.parametrize("lfs", [False, True])
def test_asset_verification_uses_requested_commit_instead_of_head(
    tmp_path: Path, lfs: bool
) -> None:
    job = prepare(tmp_path, ["a"])
    if lfs:
        job, payload, _ = pin_lfs(job)
    else:
        payload = job.tokenizer_file.read_bytes()
    changed = payload.replace(b'"a"', b'"c"')
    assert changed != payload and len(changed) == len(payload)
    job.tokenizer_file.write_bytes(lfs_pointer(changed) if lfs else changed)
    head = commit_asset(job.tokenizer_file)
    job.tokenizer_file.write_bytes(changed)
    with pytest.raises(ValueError, match="differ"):
        run_job(job)
    assert not rd.collection_path(job.data_store, ROOT, OUTPUT).exists()
    assert load_asset(job.tokenizer_file, head)[0] == changed


@pytest.mark.parametrize(
    "origin",
    [
        "git@github.com:example/assets.git",
        "ssh://git@github.com/example/assets",
        "https://fixture-user:fixture-password@github.com/example/assets.git?token=fixture-secret",
    ],
)
def test_asset_identity_does_not_expand_rewrites_or_store_credentials(
    tmp_path: Path, origin: str
) -> None:
    job = prepare(tmp_path, ["a"])
    git(job.tokenizer_file.parent, "remote", "set-url", "origin", origin)
    git(
        job.tokenizer_file.parent,
        "config",
        "url.https://fixture-rewrite@mirror.invalid/.insteadOf",
        "https://github.com/",
    )
    _, identity = load_asset(job.tokenizer_file, job.tokenizer_revision)
    assert identity["repository"] == "https://github.com/example/assets.git"


def test_empty_text_preserves_rows_with_boundaries(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["", "   "])
    result = run_job(replace(job, tasks=1, workers=1))
    assert result.row_count == 2
    assert [row.tolist() for row in sequences(job)] == [[1, 2], [1, 2]]


def test_explicit_aligned_input_overrides_root(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["a", "a"])
    aligned_ref = "data/aligned-text@v1"
    source = rd.RootParquetReader(job.data_store, ROOT)
    writer = rd.RootWriter(
        job.data_store,
        ROOT,
        data=aligned_ref,
        source=source,
        schema=pa.schema([("text", pa.string())]),
    )
    executor = LocalPipelineExecutor(
        [source, AlignedText(), writer],
        tasks=1,
        workers=1,
        logging_dir=str(tmp_path / "aligned-logs"),
    )
    writer.prepare(executor)
    executor.run()
    writer.finalize(executor)

    run_job(replace(job, input_ref=aligned_ref, tasks=1, workers=1))
    assert [row.tolist() for row in sequences(job)] == [[1, 4, 2], [1, 4, 2]]


def test_wrong_root_and_missing_asset_fail_without_token_output(tmp_path: Path) -> None:
    job = prepare(tmp_path, ["a"])
    with pytest.raises(FileNotFoundError):
        run_job(replace(job, tokenizer_file=tmp_path / "missing.json"))
    with pytest.raises((ValueError, FileNotFoundError)):
        run_job(replace(job, root_ref="data/missing-root@v1"))
    assert not rd.collection_path(job.data_store, ROOT, OUTPUT).exists()


def test_cli_help_defaults_and_invalid_resources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    help_text = subprocess.check_output(
        [sys.executable, "-m", "tokenization_scripts.tokenize", "--help"],
        text=True,
    )
    assert "UTF-8 bytes" in help_text and "Rayon threads" in help_text
    arguments = [
        "--data-store",
        str(tmp_path),
        "--root-ref",
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
    assert (job.root_ref, job.input_ref, job.output_ref) == (ROOT, ROOT, OUTPUT)
    assert get_args([*arguments, "--input-ref", ROOT]) == job
    short_arguments = [
        value.split("/", 1)[1] if value in (ROOT, OUTPUT) else value
        for value in arguments
    ]
    assert get_args(short_arguments) == job
    assert (
        replace(
            job,
            root_ref=ROOT.split("/", 1)[1],
            input_ref=ROOT.split("/", 1)[1],
            output_ref=OUTPUT.split("/", 1)[1],
        )
        == job
    )
    aligned = "data/aligned-text@v1"
    assert get_args([*arguments, "--input-ref", aligned]).input_ref == aligned
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
    with pytest.raises(ValueError, match="distinct aligned Data"):
        replace(job, output_ref=ROOT)
    with pytest.raises(ValueError, match="distinct aligned Data"):
        replace(job, output_ref=ROOT.split("/", 1)[1])
    with pytest.raises(ValueError, match="distinct aligned Data"):
        replace(job, input_ref=OUTPUT)
    for flag in ("--root-ref", "--input-ref", "--output-ref"):
        for value in ("", "annotation/text@v1", "data/text", "data/../text@v1"):
            with pytest.raises(SystemExit):
                get_args([*arguments, flag, value])
            assert "invalid data reference" in capsys.readouterr().err
            with pytest.raises(ValueError, match="invalid data reference"):
                replace(job, **{flag[2:].replace("-", "_"): value})


@pytest.mark.parametrize("workers", [1, 2])
def test_actual_cli_defaults_input_to_root_with_selected_text_and_boundaries(
    tmp_path: Path,
    workers: int,
) -> None:
    job = prepare(tmp_path, ["a b", "b a"], text_column="body")
    # The public -m entry point transports the encoder class from __main__.
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tokenization_scripts.tokenize",
            "--local",
            "--data-store",
            str(job.data_store),
            "--root-ref",
            job.root_ref.split("/", 1)[1],
            "--output-ref",
            job.output_ref.split("/", 1)[1],
            "--text-column",
            job.text_column,
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
            str(workers),
            "--threads",
            "2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
        raise
    assert process.returncode == 0, stderr
    assert "2 rows, 6 stored tokens" in stdout
    collection = rd.ParquetCollection(job.data_store, job.root_ref, job.output_ref)
    assert [sequence.tolist() for sequence in sequences(job)] == [
        [3, 4, 2],
        [4, 3, 2],
    ]
    assert token_spec(collection).prefix_tokens == ()


def test_alignment_loss_cannot_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = prepare(tmp_path, ["a", "b"])
    encode = EncodeTokens.run

    def drop_first(self, data, rank=0, world_size=1):
        stream = encode(self, data, rank, world_size)
        next(stream)
        yield from stream

    monkeypatch.setattr(EncodeTokens, "run", drop_first)
    with pytest.raises(ValueError, match="coordinates"):
        run_job(replace(job, tasks=1, workers=1))
    assert (
        rd.read_metadata(job.data_store, ROOT, OUTPUT, require_complete=False).result
        is None
    )


def test_empty_root_completes_with_empty_token_rows(tmp_path: Path) -> None:
    job = prepare(tmp_path, [])
    tokens = run_job(replace(job, tasks=1, workers=1))
    assert tokens.result == {"row_count": 0}
    assert sequences(job) == []
