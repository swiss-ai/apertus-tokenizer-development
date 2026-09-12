#!/usr/bin/env python3
"""Validate and seal flat, directly processed STEM Megatron token trees.

Unlike validate_megatron.py, this path consumes a pipeline validation report
and flat Parquet shards, not a grouped prepared-example manifest. It verifies
every source file's ownership and row coverage through the ordinary TOKMAP
sidecars before publishing a separate direct-delivery seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens import read_token_map
from tokenizers import Tokenizer

from validate_megatron import (
    COMMIT_RE,
    STRICT_VALIDATION,
    _parquet_footer_sha256,
    _write_immutable,
    sha256_file,
    validate_index,
)

SCHEMA = "stem-direct-tokenization-run/v1"
MAX_INDEX_SEQUENCE_TOKENS = 2**31 - 1
PAIR_NAMES = frozenset({"00000_tokens.bin", "00000_tokens.idx", "00000_tokens.map"})


def _source_info(path: Path) -> tuple[str, dict]:
    with pq.ParquetFile(path) as parquet:
        rows = parquet.metadata.num_rows
        row_groups = parquet.metadata.num_row_groups
        footer = _parquet_footer_sha256(path, parquet.metadata.serialized_size)
    if rows < 1:
        raise ValueError(f"empty source shard: {path}")
    return path.name, {
        "bytes": path.stat().st_size,
        "rows": rows,
        "row_groups": row_groups,
        "footer_sha256": footer,
        "sha256": sha256_file(path),
    }


def _dump_number(name: str) -> int:
    if not name.startswith("paths_file_") or not name.endswith(".txt"):
        raise ValueError(f"unexpected dump manifest: {name}")
    value = name[len("paths_file_") : -len(".txt")]
    if not value.isdecimal() or (len(value) > 1 and value.startswith("0")):
        raise ValueError(f"invalid dump number: {name}")
    return int(value)


def _completed_dumps(metadata_root: Path, expected_dumps: int) -> dict[int, list[str]]:
    pending = list((metadata_root / "dumps").rglob("paths_file_*.txt"))
    if pending:
        raise ValueError(f"unfinished dump manifest remains: {pending[0]}")
    completed = metadata_root / "completed-dumps"
    paths = sorted(completed.glob("paths_file_*.txt"), key=lambda p: _dump_number(p.name))
    numbers = [_dump_number(path.name) for path in paths]
    if numbers != list(range(expected_dumps)):
        raise ValueError(f"completed dump numbers differ: {numbers}")
    result = {}
    for number, path in zip(numbers, paths, strict=True):
        names = path.read_text(encoding="utf-8").splitlines()
        if not names or len(names) != len(set(names)):
            raise ValueError(f"empty or duplicate paths in {path}")
        if any(Path(name).name != name or not name.endswith(".parquet") for name in names):
            raise ValueError(f"non-flat source path in {path}")
        result[number] = names
    return result


def _check_records(records: array, first: int, rows: int, map_path: Path) -> None:
    if len(records) < first + rows:
        raise ValueError(f"TOKMAP source records are short: {map_path}")
    for offset in range(rows):
        if records[first + offset] != offset:
            raise ValueError(f"TOKMAP source row coordinates differ: {map_path}")


def _validate_dump(
    number: int,
    names: list[str],
    sources: dict[str, dict],
    output_root: Path,
    durable_source_root: str,
    tokenizer_sha256: str,
    vocab_size: int,
) -> dict:
    directory = output_root / f"dump-{number}"
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"token dump is absent or a symlink: {directory}")
    observed = {path.name for path in directory.iterdir()}
    if observed != PAIR_NAMES or any(path.is_symlink() for path in directory.iterdir()):
        raise ValueError(f"token dump files differ: {directory}: {sorted(observed)}")
    base = directory / "00000_tokens"
    bin_path, idx_path, map_path = (base.with_suffix(f".{suffix}") for suffix in ("bin", "idx", "map"))
    index = validate_index(idx_path, MAX_INDEX_SEQUENCE_TOKENS, STRICT_VALIDATION)
    if bin_path.stat().st_size != index["token_count"] * index["token_bytes"]:
        raise ValueError(f"token binary/index size mismatch: {bin_path}")
    idx_sha256 = sha256_file(idx_path)
    token_map = read_token_map(map_path.read_bytes())
    manifest, records = token_map["manifest"], token_map["records"]
    expected_fields = {
        "sequence_count": index["sequence_count"],
        "token_count": index["token_count"],
        "index_bytes": idx_path.stat().st_size,
        "index_sha256": idx_sha256,
        "text_column": "text",
        "id_column": "source_key",
        "output_prefix": base.name,
        "raw_dataset_root": durable_source_root,
        "skipped_rows": 0,
        "emitted_rows": index["sequence_count"],
        "task_rank": 0,
        "task_world_size": 1,
    }
    for key, expected in expected_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(f"TOKMAP {key} mismatch in {map_path}")
    expected_tokenizer = {
        "sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
        "token_size": index["token_bytes"],
        "post_processor": {
            "template": "<BOS> $A <EOS>",
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
    }
    for key, expected in expected_tokenizer.items():
        if manifest.get("tokenizer", {}).get(key) != expected:
            raise ValueError(f"TOKMAP tokenizer {key} mismatch in {map_path}")
    if not isinstance(records, array) or len(records) != index["sequence_count"]:
        raise ValueError(f"TOKMAP record count differs: {map_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or [item.get("path") for item in entries] != names:
        raise ValueError(f"TOKMAP and completed dump inventory differ: {map_path}")
    next_sequence = 0
    for entry in entries:
        name = entry["path"]
        source = sources.get(name)
        if source is None:
            raise ValueError(f"unlisted source file in TOKMAP: {name}")
        expected = {
            "bytes": source["bytes"],
            "num_rows": source["rows"],
            "num_row_groups": source["row_groups"],
            "footer_sha256": source["footer_sha256"],
            "emitted_rows": source["rows"],
            "emitted_sequences": source["rows"],
            "first_sequence": next_sequence,
        }
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f"TOKMAP {key} differs for {name}")
        _check_records(records, next_sequence, source["rows"], map_path)
        next_sequence += source["rows"]
    if next_sequence != index["sequence_count"]:
        raise ValueError(f"TOKMAP source coverage differs: {map_path}")
    return {
        "dump": number,
        "sources": names,
        "sequences": index["sequence_count"],
        "tokens": index["token_count"],
        "min_sequence_tokens": index["min_sequence_tokens"],
        "max_sequence_tokens": index["max_sequence_tokens"],
        "bin_bytes": bin_path.stat().st_size,
        "idx_bytes": idx_path.stat().st_size,
        "map_bytes": map_path.stat().st_size,
        "bin_sha256": sha256_file(bin_path),
        "idx_sha256": idx_sha256,
        "map_sha256": sha256_file(map_path),
    }


def validate(args: argparse.Namespace) -> dict:
    if args.expected_dumps < 1 or args.workers < 1:
        raise ValueError("expected dumps and workers must be positive")
    if not COMMIT_RE.fullmatch(args.implementation_commit):
        raise ValueError("implementation commit must be a full Git commit")
    if not COMMIT_RE.fullmatch(args.validator_commit):
        raise ValueError("validator commit must be a full Git commit")
    dataset_root = Path(args.dataset).resolve()
    metadata_root = Path(args.metadata_root).resolve()
    output_root = Path(args.output_folder).resolve()
    processing_report = Path(args.processing_report).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    config_path = Path(args.config).resolve()
    for path in (dataset_root, metadata_root, output_root, processing_report, tokenizer_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if (output_root / "_SUCCESS.json").exists():
        raise FileExistsError("tokenization is already sealed")
    report = json.loads(processing_report.read_text(encoding="utf-8"))
    if (report.get("identity") or report.get("dataset")) != args.dataset_name:
        raise ValueError("processing report identity differs")
    processed_rows = report.get("counts", {}).get("processed_rows")
    if not isinstance(processed_rows, int) or processed_rows < 1:
        raise ValueError("processing report has no positive processed row count")
    source_paths = sorted(dataset_root.glob("*.parquet"))
    if not source_paths or any(path.is_symlink() for path in source_paths):
        raise ValueError("processed Parquet inventory is absent or symlinked")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        source_pairs = list(executor.map(_source_info, source_paths))
    sources = dict(source_pairs)
    if len(sources) != len(source_pairs) or sum(item["rows"] for item in sources.values()) != processed_rows:
        raise ValueError("source inventory and processing row count differ")
    dumps = _completed_dumps(metadata_root, args.expected_dumps)
    all_names = [name for names in dumps.values() for name in names]
    if len(all_names) != len(set(all_names)) or set(all_names) != set(sources):
        raise ValueError("completed dumps do not cover sources exactly once")
    observed_dumps = {path.name for path in output_root.iterdir()}
    expected_dump_dirs = {f"dump-{number}" for number in dumps}
    if observed_dumps != expected_dump_dirs:
        raise ValueError("token output dump inventory differs")
    tokenizer_sha256 = sha256_file(tokenizer_path)
    vocab_size = Tokenizer.from_file(str(tokenizer_path)).get_vocab_size()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _validate_dump,
                number,
                names,
                sources,
                output_root,
                args.durable_source_root,
                tokenizer_sha256,
                vocab_size,
            )
            for number, names in dumps.items()
        ]
        pairs = [future.result() for future in futures]
    pairs.sort(key=lambda pair: pair["dump"])
    sequences = sum(pair["sequences"] for pair in pairs)
    if sequences != processed_rows:
        raise ValueError("tokenized sequences and processing rows differ")
    source_inventory = b"".join(
        json.dumps({"path": name, **sources[name]}, sort_keys=True).encode() + b"\n"
        for name in sorted(sources)
    )
    pair_payload = b"".join(
        json.dumps(pair, sort_keys=True).encode() + b"\n" for pair in pairs
    )
    run = {
        "schema_version": SCHEMA,
        "complete": True,
        "dataset": args.dataset_name,
        "dumps": len(pairs),
        "source_files": len(sources),
        "sequences": sequences,
        "tokens": sum(pair["tokens"] for pair in pairs),
        "bin_bytes": sum(pair["bin_bytes"] for pair in pairs),
        "max_sequence_tokens": max(pair["max_sequence_tokens"] for pair in pairs),
        "pins": {
            "implementation_commit": args.implementation_commit,
            "validator_commit": args.validator_commit,
            "config_sha256": sha256_file(config_path),
            "processing_report_sha256": sha256_file(processing_report),
            "tokenizer_sha256": tokenizer_sha256,
            "source_inventory_sha256": hashlib.sha256(source_inventory).hexdigest(),
            "tokenization_manifest_sha256": hashlib.sha256(pair_payload).hexdigest(),
        },
    }
    summary = (json.dumps(run, sort_keys=True, indent=2) + "\n").encode()
    _write_immutable(output_root / "TOKENIZATION_MANIFEST.jsonl", pair_payload)
    _write_immutable(output_root / "TOKENIZATION_RUN.json", summary)
    _write_immutable(output_root / "_SUCCESS.json", summary)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--processing-report", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--durable-source-root", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--expected-dumps", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(validate(args), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
