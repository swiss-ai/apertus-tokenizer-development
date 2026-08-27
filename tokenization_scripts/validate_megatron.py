#!/usr/bin/env python3
"""Validate and seal a grouped Megatron tokenization artifact.

The validator deliberately consumes the ``TOKMAP`` sidecars written by the normal
tokenization pipeline.  It does not create a second provenance format: each map is
checked against its index and the sealed prepared-Parquet inventory, while the index
is checked against the binary token file and the configured sequence limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens import read_token_map

INDEX_HEADER = b"MMIDIDX\x00\x00"
RUN_SCHEMA = "megatron-tokenization-run/v1"
CATEGORY_SCHEMA = "megatron-tokenization-category-counts/v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RANK_RE = re.compile(r"^(?P<rank>[0-9]{5})_tokens$")
PAIR_SUFFIXES = ("bin", "idx", "map")
BLOCK_BYTES = 8 * 1024 * 1024
PARQUET_TRAILER_BYTES = 8
_PAIR_CONTEXT: tuple[Any, ...] | None = None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} is not a string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path.as_posix()


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} is not an integer")
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{label} is out of range: {value}")
    return value


def load_prepared_inventory(
    dataset_root: Path,
    manifest_path: Path,
    marker_path: Path,
    expected_categories: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Load the exact sealed examples inventory and validate its release pins."""

    if not marker_path.is_file():
        raise FileNotFoundError(f"prepared-dataset marker is missing: {marker_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared examples manifest is missing: {manifest_path}"
        )
    marker = _read_json_object(marker_path)
    if marker.get("complete") is not True or marker.get("smoke") is not False:
        raise ValueError("prepared-dataset marker is not a complete production seal")
    manifest_sha256 = sha256_file(manifest_path)
    if marker.get("pins", {}).get("examples_manifest_sha256") != manifest_sha256:
        raise ValueError(
            "prepared examples manifest differs from its completion marker"
        )

    inventory: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"prepared manifest line {line_number} is not an object")
        relative = _safe_relative_path(
            row.get("relative_path"), label=f"manifest path on line {line_number}"
        )
        if relative in inventory:
            raise ValueError(f"duplicate prepared manifest path: {relative}")
        category = row.get("category")
        if row.get("kind") != "examples" or category not in expected_categories:
            raise ValueError(f"invalid examples manifest entry: {relative}")
        if PurePosixPath(relative).parts[:2] != ("examples", category):
            raise ValueError(f"prepared path/category mismatch: {relative}")
        rows = _positive_int(row.get("rows"), label=f"rows for {relative}")
        file_bytes = _positive_int(
            row.get("file_bytes"), label=f"file bytes for {relative}"
        )
        digest = row.get("sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid prepared digest for {relative}")
        path = dataset_root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.stat().st_size != file_bytes:
            raise ValueError(f"prepared file size differs from manifest: {relative}")
        inventory[relative] = {
            "category": category,
            "rows": rows,
            "bytes": file_bytes,
            "sha256": digest,
        }
    if not inventory:
        raise ValueError("prepared examples manifest is empty")
    observed_categories = {entry["category"] for entry in inventory.values()}
    if observed_categories != expected_categories:
        raise ValueError(
            "prepared categories differ: "
            f"expected {sorted(expected_categories)}, found {sorted(observed_categories)}"
        )
    return inventory, {
        "marker_sha256": sha256_file(marker_path),
        "manifest_sha256": manifest_sha256,
    }


def discover_pairs(
    output_root: Path, expected_categories: set[str]
) -> list[tuple[str, str]]:
    """Return ``(relative prefix, category)`` pairs after exact triple discovery."""

    entries = sorted(output_root.rglob("*"))
    partials = [path for path in entries if ".partial" in path.name]
    if partials:
        raise ValueError(f"partial token outputs remain, first: {partials[0]}")
    for path in entries:
        if path.is_symlink():
            raise ValueError(f"token output contains a symlink: {path}")
        if path.is_file() and path.suffix.removeprefix(".") not in PAIR_SUFFIXES:
            raise ValueError(f"unexpected file in unsealed token output: {path}")
    by_suffix: dict[str, set[str]] = {suffix: set() for suffix in PAIR_SUFFIXES}
    for suffix in PAIR_SUFFIXES:
        for path in output_root.rglob(f"*.{suffix}"):
            relative = path.relative_to(output_root)
            if len(relative.parts) < 3:
                raise ValueError(f"token pair is outside a category/dump: {relative}")
            category = relative.parts[0]
            if category not in expected_categories:
                raise ValueError(f"unexpected token category {category!r}: {relative}")
            by_suffix[suffix].add(relative.with_suffix("").as_posix())
    prefixes = set.union(*by_suffix.values())
    if not prefixes:
        raise ValueError(f"no Megatron token pairs found under {output_root}")
    for prefix in sorted(prefixes):
        missing = [
            suffix for suffix in PAIR_SUFFIXES if prefix not in by_suffix[suffix]
        ]
        if missing:
            raise ValueError(
                f"incomplete token pair {prefix}: missing {', '.join(missing)}"
            )
    observed_categories = {PurePosixPath(prefix).parts[0] for prefix in prefixes}
    if observed_categories != expected_categories:
        raise ValueError(
            "token categories differ: "
            f"expected {sorted(expected_categories)}, found {sorted(observed_categories)}"
        )
    return [(prefix, PurePosixPath(prefix).parts[0]) for prefix in sorted(prefixes)]


def _array_all_equal(actual: np.ndarray, expected) -> bool:
    """Compare a memmapped array in bounded chunks."""

    chunk_items = 1_000_000
    for start in range(0, actual.size, chunk_items):
        end = min(actual.size, start + chunk_items)
        if not bool(np.array_equal(actual[start:end], expected(start, end))):
            return False
    return True


def validate_index(path: Path, max_sequence_tokens: int) -> dict[str, int]:
    """Validate one Megatron index without loading its arrays into memory."""

    with path.open("rb") as stream:
        header = stream.read(len(INDEX_HEADER))
        version_bytes = stream.read(8)
        dtype_bytes = stream.read(1)
        counts = stream.read(16)
    if header != INDEX_HEADER or len(version_bytes) != 8 or len(dtype_bytes) != 1:
        raise ValueError(f"invalid Megatron index header: {path}")
    if struct.unpack("<Q", version_bytes)[0] != 1:
        raise ValueError(f"unsupported Megatron index version: {path}")
    dtype_code = struct.unpack("<B", dtype_bytes)[0]
    if dtype_code not in (4, 8):
        raise ValueError(f"unsupported Megatron token dtype code {dtype_code}: {path}")
    if len(counts) != 16:
        raise ValueError(f"truncated Megatron index counts: {path}")
    sequence_count, document_count = struct.unpack("<QQ", counts)
    if sequence_count < 1 or document_count != sequence_count + 1:
        raise ValueError(f"invalid Megatron sequence/document counts: {path}")
    expected_bytes = 34 + 4 * sequence_count + 8 * sequence_count + 8 * document_count
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Megatron index size disagrees with its counts: {path}")

    lengths_offset = 34
    pointers_offset = lengths_offset + 4 * sequence_count
    documents_offset = pointers_offset + 8 * sequence_count
    lengths = np.memmap(
        path, mode="r", dtype="<i4", offset=lengths_offset, shape=(sequence_count,)
    )
    if int(lengths.min()) < 1 or int(lengths.max()) > max_sequence_tokens:
        raise ValueError(f"sequence length is outside 1..{max_sequence_tokens}: {path}")
    token_count = int(lengths.sum(dtype=np.int64))
    token_bytes = 4 if dtype_code == 4 else 2
    pointers = np.memmap(
        path, mode="r", dtype="<i8", offset=pointers_offset, shape=(sequence_count,)
    )
    cumulative = 0
    chunk_items = 1_000_000
    for start in range(0, sequence_count, chunk_items):
        end = min(sequence_count, start + chunk_items)
        expected = np.empty(end - start, dtype=np.int64)
        expected[0] = cumulative
        if end - start > 1:
            np.cumsum(lengths[start : end - 1], dtype=np.int64, out=expected[1:])
            expected[1:] = (expected[1:] + cumulative) * token_bytes
        expected[0] *= token_bytes
        if not bool(np.array_equal(pointers[start:end], expected)):
            raise ValueError(f"Megatron sequence pointers are not cumulative: {path}")
        cumulative += int(lengths[start:end].sum(dtype=np.int64))
    documents = np.memmap(
        path, mode="r", dtype="<i8", offset=documents_offset, shape=(document_count,)
    )
    if not _array_all_equal(
        documents, lambda start, end: np.arange(start, end, dtype=np.int64)
    ):
        raise ValueError(f"Megatron document indices are not 0..N: {path}")
    return {
        "sequence_count": int(sequence_count),
        "token_count": token_count,
        "token_bytes": token_bytes,
        "max_sequence_tokens": int(lengths.max()),
        "min_sequence_tokens": int(lengths.min()),
    }


def _normalized_source_path(value: Any) -> str:
    relative = _safe_relative_path(value, label="token-map source path")
    # Prepared manifests include the artifact kind while ProvenanceParquetReader is
    # rooted at the sealed release and therefore normally emits the same path.  Keep
    # this check explicit instead of guessing alternative layouts.
    if not relative.startswith("examples/"):
        raise ValueError(f"token-map source is not prepared examples: {relative}")
    return relative


def _parquet_footer_sha256(path: Path, serialized_size: int) -> str:
    size = path.stat().st_size
    if serialized_size < 0 or size < serialized_size + PARQUET_TRAILER_BYTES:
        raise ValueError(f"invalid Parquet footer size: {path}")
    with path.open("rb") as stream:
        stream.seek(size - serialized_size - PARQUET_TRAILER_BYTES)
        footer = stream.read(serialized_size + PARQUET_TRAILER_BYTES)
    if len(footer) != serialized_size + PARQUET_TRAILER_BYTES:
        raise ValueError(f"short Parquet footer read: {path}")
    return hashlib.sha256(footer).hexdigest()


def validate_pair(
    output_root_value: str,
    prefix: str,
    category: str,
    dataset_root_value: str,
    expected_sources: dict[str, dict[str, Any]],
    tokenizer_sha256: str,
    tokenizer_vocab_size: int,
    max_sequence_tokens: int,
    text_column: str,
    id_column: str,
) -> dict[str, Any]:
    output_root = Path(output_root_value)
    dataset_root = Path(dataset_root_value)
    base = output_root.joinpath(*PurePosixPath(prefix).parts)
    bin_path = base.with_suffix(".bin")
    idx_path = base.with_suffix(".idx")
    map_path = base.with_suffix(".map")
    index = validate_index(idx_path, max_sequence_tokens)
    if bin_path.stat().st_size != index["token_count"] * index["token_bytes"]:
        raise ValueError(f"token binary size disagrees with index: {bin_path}")
    index_sha256 = sha256_file(idx_path)
    token_map = read_token_map(map_path.read_bytes())
    manifest = token_map["manifest"]
    for key, expected in (
        ("sequence_count", index["sequence_count"]),
        ("token_count", index["token_count"]),
        ("index_bytes", idx_path.stat().st_size),
        ("index_sha256", index_sha256),
        ("text_column", text_column),
        ("id_column", id_column),
        ("output_prefix", base.name),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"token map differs for {key}: {map_path}")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise TypeError(f"token map has no tokenizer identity: {map_path}")
    expected_tokenizer = {
        "sha256": tokenizer_sha256,
        "vocab_size": tokenizer_vocab_size,
        "token_size": index["token_bytes"],
    }
    for key, expected in expected_tokenizer.items():
        if tokenizer.get(key) != expected:
            raise ValueError(f"token map tokenizer differs for {key}: {map_path}")
    if tokenizer.get("post_processor") != {
        "template": "<BOS> $A <EOS>",
        "bos_token_id": 1,
        "eos_token_id": 2,
    }:
        raise ValueError(f"token map has the wrong postprocessor: {map_path}")
    source_root = manifest.get("raw_dataset_root")
    if (
        not isinstance(source_root, str)
        or Path(source_root).resolve() != dataset_root.resolve()
    ):
        raise ValueError(f"token map has the wrong prepared root: {map_path}")
    if (
        manifest.get("skipped_rows") != 0
        or manifest.get("emitted_rows") != index["sequence_count"]
    ):
        raise ValueError(f"prepared example rows were skipped or repeated: {map_path}")
    rank_match = RANK_RE.fullmatch(base.name)
    rank = manifest.get("task_rank")
    world_size = manifest.get("task_world_size")
    if (
        rank_match is None
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or not 0 <= rank < world_size
        or rank != int(rank_match.group("rank"))
    ):
        raise ValueError(f"token map has invalid task ownership: {map_path}")

    sources = []
    for entry in manifest.get("files", []):
        relative = _normalized_source_path(entry.get("path"))
        expected = expected_sources.get(relative)
        if expected is None:
            raise ValueError(f"token map names an unsealed prepared file: {relative}")
        if expected["category"] != category:
            raise ValueError(f"token pair crosses categories: {map_path}")
        if (
            entry.get("bytes") != expected["bytes"]
            or entry.get("num_rows") != expected["rows"]
        ):
            raise ValueError(f"token-map source shape differs: {relative}")
        if (
            entry.get("emitted_rows") != expected["rows"]
            or entry.get("emitted_sequences") != expected["rows"]
        ):
            raise ValueError(f"token-map source coverage differs: {relative}")
        source_path = dataset_root.joinpath(*PurePosixPath(relative).parts)
        with pq.ParquetFile(source_path) as parquet:
            if parquet.metadata.num_rows != expected["rows"]:
                raise ValueError(f"prepared Parquet row count changed: {relative}")
            if entry.get("num_row_groups") != parquet.metadata.num_row_groups:
                raise ValueError(f"prepared Parquet row groups changed: {relative}")
            serialized_size = parquet.metadata.serialized_size
        if entry.get("footer_sha256") != _parquet_footer_sha256(
            source_path, serialized_size
        ):
            raise ValueError(f"prepared Parquet footer changed: {relative}")
        sources.append(relative)
    if not sources:
        raise ValueError(f"token pair has no prepared source files: {map_path}")

    return {
        "relative_prefix": prefix,
        "category": category,
        "rank": rank,
        "world_size": world_size,
        "sequences": index["sequence_count"],
        "tokens": index["token_count"],
        "min_sequence_tokens": index["min_sequence_tokens"],
        "max_sequence_tokens": index["max_sequence_tokens"],
        "bin_bytes": bin_path.stat().st_size,
        "idx_bytes": idx_path.stat().st_size,
        "map_bytes": map_path.stat().st_size,
        "bin_sha256": sha256_file(bin_path),
        "idx_sha256": index_sha256,
        "map_sha256": sha256_file(map_path),
        "sources": sources,
    }


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace immutable artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialize_pair_worker(*values: Any) -> None:
    global _PAIR_CONTEXT
    _PAIR_CONTEXT = values


def _validate_discovered_pair(pair: tuple[str, str]) -> dict[str, Any]:
    if _PAIR_CONTEXT is None:
        raise RuntimeError("pair-validation worker was not initialized")
    return validate_pair(
        _PAIR_CONTEXT[0],
        pair[0],
        pair[1],
        *_PAIR_CONTEXT[1:],
    )


def validate_and_seal(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_folder).resolve()
    dataset_root = Path(args.dataset).resolve()
    manifest_path = Path(args.manifest).resolve()
    marker_path = Path(args.dataset_marker).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    config_path = Path(args.config).resolve()
    expected_categories = {
        value.strip() for value in args.expected_categories.split(",") if value.strip()
    }
    if not expected_categories:
        raise ValueError("expected categories are empty")
    if not COMMIT_RE.fullmatch(args.implementation_commit):
        raise ValueError("implementation commit must be a full lowercase Git commit")
    if args.workers < 1 or args.max_sequence_tokens < 1:
        raise ValueError("workers and max sequence tokens must be positive")
    for path, label in (
        (output_root, "token output root"),
        (dataset_root, "prepared dataset root"),
        (tokenizer_path, "tokenizer"),
        (config_path, "tokenization config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    for marker_name in (
        "TOKENIZATION_MANIFEST.jsonl",
        "TOKENIZATION_RUN.json",
        "CATEGORY_COUNTS.json",
        "_SUCCESS.json",
    ):
        if (output_root / marker_name).exists():
            raise FileExistsError(
                f"refusing to reseal tokenization: {output_root / marker_name}"
            )

    prepared, prepared_pins = load_prepared_inventory(
        dataset_root, manifest_path, marker_path, expected_categories
    )
    pairs = discover_pairs(output_root, expected_categories)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    from tokenizers import Tokenizer

    tokenizer_vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    worker_context = (
        str(output_root),
        str(dataset_root),
        prepared,
        tokenizer_sha256,
        tokenizer_vocab_size,
        args.max_sequence_tokens,
        args.text_column,
        args.id_column,
    )
    if args.workers == 1:
        reports = [
            validate_pair(
                str(output_root),
                prefix,
                category,
                *worker_context[1:],
            )
            for prefix, category in pairs
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_pair_worker,
            initargs=worker_context,
        ) as executor:
            reports = list(executor.map(_validate_discovered_pair, pairs))

    source_owners: dict[str, str] = {}
    for report in reports:
        for source in report["sources"]:
            if source in source_owners:
                raise ValueError(
                    f"prepared file is tokenized more than once: {source} in "
                    f"{source_owners[source]} and {report['relative_prefix']}"
                )
            source_owners[source] = report["relative_prefix"]
    missing = sorted(set(prepared) - set(source_owners))
    unexpected = sorted(set(source_owners) - set(prepared))
    if missing or unexpected:
        raise ValueError(
            "token-map source inventory differs from prepared manifest; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    by_dump: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        dump = PurePosixPath(report["relative_prefix"]).parent.as_posix()
        by_dump.setdefault(dump, []).append(report)
    for dump, dump_reports in by_dump.items():
        world_sizes = {report["world_size"] for report in dump_reports}
        if len(world_sizes) != 1:
            raise ValueError(f"tokenizer task world sizes differ in {dump}")
        world_size = world_sizes.pop()
        ranks = {report["rank"] for report in dump_reports}
        if ranks != set(range(world_size)):
            raise ValueError(
                f"tokenizer task ranks are incomplete in {dump}: "
                f"expected 0..{world_size - 1}, found {sorted(ranks)[:10]}"
            )

    category_counts: dict[str, dict[str, int]] = {}
    totals: Counter[str] = Counter()
    for category in sorted(expected_categories):
        selected = [report for report in reports if report["category"] == category]
        counts = {
            "pairs": len(selected),
            "source_files": sum(len(report["sources"]) for report in selected),
            "sequences": sum(report["sequences"] for report in selected),
            "tokens": sum(report["tokens"] for report in selected),
            "bin_bytes": sum(report["bin_bytes"] for report in selected),
            "idx_bytes": sum(report["idx_bytes"] for report in selected),
            "map_bytes": sum(report["map_bytes"] for report in selected),
            "min_sequence_tokens": min(
                report["min_sequence_tokens"] for report in selected
            ),
            "max_sequence_tokens": max(
                report["max_sequence_tokens"] for report in selected
            ),
        }
        category_counts[category] = counts
        for key in (
            "pairs",
            "source_files",
            "sequences",
            "tokens",
            "bin_bytes",
            "idx_bytes",
            "map_bytes",
        ):
            totals[key] += counts[key]
    totals["min_sequence_tokens"] = min(
        report["min_sequence_tokens"] for report in reports
    )
    totals["max_sequence_tokens"] = max(
        report["max_sequence_tokens"] for report in reports
    )

    pair_inventory = [
        {
            key: report[key]
            for key in (
                "relative_prefix",
                "category",
                "rank",
                "world_size",
                "sequences",
                "tokens",
                "bin_bytes",
                "idx_bytes",
                "map_bytes",
                "bin_sha256",
                "idx_sha256",
                "map_sha256",
                "sources",
            )
        }
        for report in reports
    ]
    pair_inventory_payload = b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in pair_inventory
    )
    pair_inventory_sha256 = hashlib.sha256(pair_inventory_payload).hexdigest()
    category_document = {
        "schema_version": CATEGORY_SCHEMA,
        "categories": category_counts,
        "totals": dict(totals),
    }
    run_document = {
        "schema_version": RUN_SCHEMA,
        "complete": True,
        "dataset": args.dataset_name,
        "tokenizer_name": args.tokenizer_name,
        "constraints": {
            "max_sequence_tokens": args.max_sequence_tokens,
            "expected_categories": sorted(expected_categories),
            "text_column": args.text_column,
            "id_column": args.id_column,
        },
        "pins": {
            "implementation_commit": args.implementation_commit,
            "config_sha256": sha256_file(config_path),
            "prepared_marker_sha256": prepared_pins["marker_sha256"],
            "prepared_examples_manifest_sha256": prepared_pins["manifest_sha256"],
            "tokenizer_sha256": tokenizer_sha256,
            "tokenization_manifest_sha256": pair_inventory_sha256,
        },
        "prepared": {
            "files": len(prepared),
            "rows": sum(entry["rows"] for entry in prepared.values()),
            "bytes": sum(entry["bytes"] for entry in prepared.values()),
        },
        "totals": dict(totals),
    }
    category_payload = (
        json.dumps(category_document, indent=2, sort_keys=True).encode() + b"\n"
    )
    run_payload = json.dumps(run_document, indent=2, sort_keys=True).encode() + b"\n"
    _write_immutable(
        output_root / "TOKENIZATION_MANIFEST.jsonl", pair_inventory_payload
    )
    _write_immutable(output_root / "CATEGORY_COUNTS.json", category_payload)
    _write_immutable(output_root / "TOKENIZATION_RUN.json", run_payload)
    # Completion is deliberately the final publication and byte-identical to the
    # run summary, mirroring the prepared-text artifact's marker-last contract.
    _write_immutable(output_root / "_SUCCESS.json", run_payload)
    return run_document


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-marker", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--expected-categories", required=True)
    parser.add_argument("--max-sequence-tokens", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = validate_and_seal(get_args())
    print(json.dumps(result, indent=2, sort_keys=True))
