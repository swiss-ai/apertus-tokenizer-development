#!/usr/bin/env python3
"""Build a complete token derivative from sealed FineTranslations row repairs.

Copy inherited sequences byte for byte; encode only source-proven changed rows.
Use the owning pipeline's encoder, index/map serializers and sequence copier.
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens.megatron_index import (
    pack_megatron_index,
    read_megatron_index,
)
from data_pipeline_pretrain.pipeline.tokens.megatron_tokenizer import (
    MegatronSequenceEncoder,
)
from data_pipeline_pretrain.pipeline.tokens.pair_splitter import _copy_sequences
from data_pipeline_pretrain.pipeline.tokens.token_map import (
    pack_token_map,
    read_token_map,
)
from data_pipeline_pretrain.utils.hashing import sha256_file

from tokenization_scripts.validate_megatron import _parquet_footer_sha256

EXTENSIONS = (".bin", ".idx", ".map")
SCHEMA = "finetranslations-source-token-repair/v1"
_CONTEXT = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("unsafe source-relative path")
    return path.as_posix()


def discover(root):
    sets = {extension: set() for extension in EXTENSIONS}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("token root contains a symlink")
        if path.is_file() and path.suffix in sets:
            sets[path.suffix].add(path.relative_to(root).with_suffix("").as_posix())
    if not sets[".bin"] or any(value != sets[".bin"] for value in sets.values()):
        raise ValueError("empty or incomplete original token inventory")
    if any(name.split("/")[0] == "_reserved" for name in sets[".bin"]):
        raise ValueError("consume the complete original, unsplit token root")
    return sorted(sets[".bin"], key=lambda value: value.encode())


def identities(prefix):
    return {
        extension: {
            "bytes": prefix.with_suffix(extension).stat().st_size,
            "sha256": sha256_file(prefix.with_suffix(extension)),
        }
        for extension in EXTENSIONS
    }


def shape(path):
    metadata = pq.read_metadata(path)
    return {
        "bytes": path.stat().st_size,
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "footer_sha256": _parquet_footer_sha256(path, metadata.serialized_size),
    }


def prepared_identity(task):
    old_root, new_root, report = task
    name = safe_path(report["path"])
    result = {}
    for kind, root in (("old", old_root), ("new", new_root)):
        path = Path(root) / name
        if path.resolve(strict=True) != path.absolute():
            raise ValueError("prepared input must have no symlink components")
        expected = report["input_sha256" if kind == "old" else "output_sha256"]
        if sha256_file(path) != expected:
            raise ValueError("prepared file differs from sealed row-repair evidence")
        result[kind] = shape(path)
        if result[kind]["num_rows"] != report["rows"]:
            raise ValueError("prepared population changed")
    return name, result


def load_prepared(new_root, workers):
    seal_path, report_path = (
        new_root / "_SUCCESS.json",
        new_root / "PROCESSING_REPORT.json",
    )
    seal, report = (json.loads(path.read_text()) for path in (seal_path, report_path))
    report_sha = sha256_file(report_path)
    if (
        seal.get("complete") is not True
        or seal.get("processing_report_sha256") != report_sha
        or report.get("schema") != "finetranslations-source-pii-repair/v1"
        or seal.get("producer_commit") != report.get("producer_commit")
        or seal.get("dataset_id") != report.get("dataset_id")
    ):
        raise ValueError("prepared row repair is not a complete pinned release")
    old_root = Path(report["source_root"]).resolve(strict=True)
    files, changed = {}, {}
    for item in report["reports"]:
        name = safe_path(item["path"])
        if name in files:
            raise ValueError("duplicate prepared file")
        files[name] = item
        changed[name] = {}
        for proof in item["changed"]:
            row = proof["prepared_row"]
            if (
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < item["rows"]
                or row in changed[name]
                or proof.get("changed") is not True
                or proof.get("full_historical_replay_exact") is not True
            ):
                raise ValueError("invalid source-proven changed-row inventory")
            changed[name][row] = proof
        if len(changed[name]) != item["changed_rows"]:
            raise ValueError("changed-row accounting differs")
    expected = sorted(files)
    if sorted(report["files"]) != expected:
        raise ValueError("prepared manifest inventory differs")
    for root in (old_root, new_root):
        actual = sorted(
            path.relative_to(root).as_posix() for path in root.rglob("*.parquet")
        )
        if actual != expected:
            raise ValueError("prepared complete file population differs")
    if (
        sum(item["rows"] for item in files.values()) != report["rows"]
        or sum(len(rows) for rows in changed.values()) != report["changed_rows"]
        or seal.get("rows") != report["rows"]
        or seal.get("changed_rows") != report["changed_rows"]
    ):
        raise ValueError("prepared root accounting differs")
    tasks = [(str(old_root), str(new_root), item) for item in files.values()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shapes = dict(pool.map(prepared_identity, tasks))
    pins = {
        "seal_sha256": sha256_file(seal_path),
        "report_sha256": report_sha,
        "producer_commit": report["producer_commit"],
        "dataset_id": report["dataset_id"],
    }
    return old_root, files, changed, shapes, pins


def rows_at(path, wanted):
    result, offset = {}, 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=512, columns=["id", "text"], use_threads=False
    ):
        selected = [row for row in wanted if offset <= row < offset + len(batch)]
        for row in selected:
            result[row] = batch.slice(row - offset, 1).to_pylist()[0]
        offset += len(batch)
    if set(result) != set(wanted):
        raise ValueError("prepared row is missing")
    return result


def initialize(context):
    global _CONTEXT
    _CONTEXT = context


def verify_completed_pair(task):
    source_root, output_root, report = task
    for root, key in ((source_root, "input"), (output_root, "output")):
        if identities(Path(root) / report["prefix"]) != report[key]:
            raise ValueError("completed input/output payload changed before root seal")


def repair_pair(relative):
    context = _CONTEXT
    source = Path(context["source_root"]) / relative
    target = Path(context["output_root"]) / relative
    before = identities(source)
    index = read_megatron_index(source.with_suffix(".idx"), before[".bin"]["bytes"])
    token_map = read_token_map(source.with_suffix(".map").read_bytes())
    manifest, records = token_map["manifest"], token_map["records"]
    if (
        index.dtype_code != 4
        or manifest["token_count"] != index.token_count
        or manifest["sequence_count"] != index.sequence_count
        or manifest.get("index_bytes") != before[".idx"]["bytes"]
        or manifest.get("index_sha256") != before[".idx"]["sha256"]
        or Path(manifest["raw_dataset_root"]).resolve(strict=True)
        != Path(context["old_root"])
        or manifest.get("tokenizer", {}).get("sha256") != context["tokenizer_sha256"]
        or manifest["tokenizer"].get("post_processor")
        != {"template": "<BOS> $A <EOS>", "bos_token_id": 1, "eos_token_id": 2}
    ):
        raise ValueError("original tokenizer, map or index provenance differs")
    replacements, covered, new_files = {}, [], []
    encoder = MegatronSequenceEncoder(context["tokenizer"])
    row_array = np.frombuffer(records, dtype="<u8")
    with source.with_suffix(".bin").open("rb") as binary:
        for entry in manifest["files"]:
            name = safe_path(entry["path"])
            if name not in context["shapes"]:
                raise ValueError("map names an undeclared prepared file")
            shapes = context["shapes"][name]
            if any(entry.get(key) != value for key, value in shapes["old"].items()):
                raise ValueError("original source map footer or shape differs")
            new_files.append({**entry, **shapes["new"]})
            start, stop = (
                entry["first_sequence"],
                entry["first_sequence"] + entry["emitted_sequences"],
            )
            source_rows = row_array[start:stop]
            wanted = {}
            for row, proof in context["changed"][name].items():
                left, right = (
                    np.searchsorted(source_rows, row, side="left"),
                    np.searchsorted(source_rows, row, side="right"),
                )
                if left != right:
                    wanted[row] = (proof, range(start + int(left), start + int(right)))
            if not wanted:
                continue
            old_rows = rows_at(Path(context["old_root"]) / name, wanted)
            new_rows = rows_at(Path(context["new_root"]) / name, wanted)
            for row, (proof, ordinals) in wanted.items():
                old, new = old_rows[row], new_rows[row]
                if (
                    old["id"] != proof["id"]
                    or new["id"] != proof["id"]
                    or hashlib.sha256(old["text"].encode()).hexdigest()
                    != proof["old_text_sha256"]
                    or hashlib.sha256(new["text"].encode()).hexdigest()
                    != proof["new_text_sha256"]
                ):
                    raise ValueError(
                        "source row differs from the prepared repair proof"
                    )
                old_tokens = np.asarray(
                    encoder.encode(old["text"]), dtype="<u4"
                ).tobytes()
                new_ids = encoder.encode(new["text"])
                if (
                    new_ids[0] != 1
                    or new_ids[-1] != 2
                    or any(token in (1, 2) for token in new_ids[1:-1])
                ):
                    raise ValueError(
                        "corrected row collides with document-boundary tokens"
                    )
                new_tokens = np.asarray(new_ids, dtype="<u4").tobytes()
                for ordinal in ordinals:
                    binary.seek(int(index.pointers[ordinal]))
                    if binary.read(int(index.lengths[ordinal]) * 4) != old_tokens:
                        raise ValueError(
                            "changed source row does not exactly reencode original tokens"
                        )
                    replacements[ordinal] = new_tokens
                covered.append([name, row])
    target.parent.mkdir(parents=True, exist_ok=True)
    lengths = index.lengths.copy()
    if not replacements:
        for extension in (".bin", ".idx"):
            os.link(source.with_suffix(extension), target.with_suffix(extension))
    else:
        with (
            source.with_suffix(".bin").open("rb") as old,
            target.with_suffix(".bin.partial").open("wb") as new,
        ):
            start = 0
            for ordinal in sorted(replacements):
                _copy_sequences(old, new, index, range(start, ordinal))
                new.write(replacements[ordinal])
                lengths[ordinal] = len(replacements[ordinal]) // 4
                start = ordinal + 1
            _copy_sequences(old, new, index, range(start, index.sequence_count))
        target.with_suffix(".idx").write_bytes(pack_megatron_index(lengths, 4))
        target.with_suffix(".bin.partial").rename(target.with_suffix(".bin"))
    output_index = read_megatron_index(
        target.with_suffix(".idx"), target.with_suffix(".bin").stat().st_size
    )
    # Compare every inherited token byte and every corrected encoding, bounded by
    # an 8 MiB block. No count-only or file-existence acceptance.
    with (
        source.with_suffix(".bin").open("rb") as old,
        target.with_suffix(".bin").open("rb") as new,
    ):
        start = 0
        for ordinal in sorted(replacements) + [index.sequence_count]:
            stop_byte = (
                int(index.pointers[ordinal])
                if ordinal < index.sequence_count
                else before[".bin"]["bytes"]
            )
            old.seek(
                int(index.pointers[start])
                if start < index.sequence_count
                else before[".bin"]["bytes"]
            )
            remaining = stop_byte - old.tell()
            while remaining:
                block = old.read(min(8 << 20, remaining))
                if not block or new.read(len(block)) != block:
                    raise ValueError("inherited token bytes changed")
                remaining -= len(block)
            if (
                ordinal < index.sequence_count
                and new.read(len(replacements[ordinal])) != replacements[ordinal]
            ):
                raise ValueError("corrected encoding differs from written tokens")
            start = ordinal + 1
        if new.read(1):
            raise ValueError("output contains extra token bytes")
    new_manifest = {
        **manifest,
        "raw_dataset_root": context["new_root"],
        "index_sha256": sha256_file(target.with_suffix(".idx")),
        "index_bytes": target.with_suffix(".idx").stat().st_size,
        "prepared_repair": context["prepared_pins"],
        "token_repair_producer_commit": context["commit"],
        "parent_pair": {
            "root": context["source_root"],
            "prefix": relative,
            "files": before,
        },
    }
    target.with_suffix(".map").write_bytes(
        pack_token_map(new_files, records, output_index.token_count, new_manifest)
    )
    output_map = read_token_map(target.with_suffix(".map").read_bytes())
    if output_map["records"] != records or identities(source) != before:
        raise ValueError("source coordinates or original input changed")
    result = {
        "prefix": relative,
        "input": before,
        "output": identities(target),
        "sequences": index.sequence_count,
        "input_tokens": index.token_count,
        "output_tokens": output_index.token_count,
        "changed_sequences": len(replacements),
        "changed_source_rows": covered,
        "all_inherited_token_bytes_equal": True,
        "all_changed_source_reencodings_exact": True,
        "all_coordinates_preserved": True,
    }
    receipt = Path(context["output_root"]) / "receipts" / (relative + ".json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt, result)
    return result


def repair_root(
    source_root,
    prepared_root,
    tokenizer,
    output_root,
    dataset_id,
    commit,
    pipeline_commit,
    workers=16,
):
    source_root, prepared_root, tokenizer = (
        Path(value).resolve(strict=True)
        for value in (source_root, prepared_root, tokenizer)
    )
    output_root = Path(output_root).resolve()
    if (
        not 1 <= workers <= 64
        or any(
            len(value) != 40 or any(c not in "0123456789abcdef" for c in value)
            for value in (commit, pipeline_commit)
        )
        or not dataset_id
    ):
        raise ValueError(
            "require dataset identity, full producer commit and 1–64 workers"
        )
    if output_root.exists():
        raise FileExistsError(output_root)
    if any(
        root == output_root or root in output_root.parents
        for root in (source_root, prepared_root)
    ):
        raise ValueError("output must be outside immutable inputs")
    old_root, files, changed, shapes, pins = load_prepared(prepared_root, workers)
    if old_root == output_root or old_root in output_root.parents:
        raise ValueError("output must be outside historical prepared input")
    pairs = discover(source_root)
    context = {
        "source_root": str(source_root),
        "old_root": str(old_root),
        "new_root": str(prepared_root),
        "output_root": str(output_root),
        "tokenizer": str(tokenizer),
        "tokenizer_sha256": sha256_file(tokenizer),
        "changed": changed,
        "shapes": shapes,
        "prepared_pins": pins,
        "commit": commit,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    spec = {
        "schema": SCHEMA,
        "dataset_id": dataset_id,
        "producer_commit": commit,
        "pipeline_commit": pipeline_commit,
        "pipeline_components": {
            name: {
                "path": str(Path(sys.modules[name].__file__).resolve()),
                "sha256": sha256_file(sys.modules[name].__file__),
            }
            for name in (
                "data_pipeline_pretrain.pipeline.tokens.megatron_index",
                "data_pipeline_pretrain.pipeline.tokens.megatron_tokenizer",
                "data_pipeline_pretrain.pipeline.tokens.pair_splitter",
                "data_pipeline_pretrain.pipeline.tokens.token_map",
            )
        },
        "source_root": str(source_root),
        "prepared_root": str(prepared_root),
        "prepared_pins": pins,
        "tokenizer_sha256": context["tokenizer_sha256"],
        "pairs": pairs,
    }
    write_json(output_root / "RUN_SPEC.json", spec)
    if workers == 1:
        initialize(context)
        reports = list(map(repair_pair, pairs))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=initialize, initargs=(context,)
        ) as pool:
            reports = list(pool.map(repair_pair, pairs))
    covered = {tuple(row) for item in reports for row in item["changed_source_rows"]}
    if covered != {(name, row) for name, rows in changed.items() for row in rows}:
        raise ValueError(
            "not every prepared changed row is represented in the full token root"
        )
    if discover(source_root) != pairs or discover(output_root) != pairs:
        raise ValueError("complete token pair population differs")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                verify_completed_pair,
                [(str(source_root), str(output_root), item) for item in reports],
            )
        )
    # Recheck every physical prepared input and both release pins before sealing.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if (
            dict(
                pool.map(
                    prepared_identity,
                    [
                        (str(old_root), str(prepared_root), item)
                        for item in files.values()
                    ],
                )
            )
            != shapes
        ):
            raise ValueError("prepared inputs changed during token repair")
    if (
        sha256_file(prepared_root / "_SUCCESS.json") != pins["seal_sha256"]
        or sha256_file(prepared_root / "PROCESSING_REPORT.json")
        != pins["report_sha256"]
        or sha256_file(tokenizer) != context["tokenizer_sha256"]
    ):
        raise ValueError("prepared release or tokenizer changed")
    if any(
        sha256_file(value["path"]) != value["sha256"]
        for value in spec["pipeline_components"].values()
    ):
        raise ValueError("pipeline dependency changed during token repair")
    manifest = {
        **spec,
        "reports": reports,
        "sequences": sum(item["sequences"] for item in reports),
        "tokens": sum(item["output_tokens"] for item in reports),
        "changed_source_rows": len(covered),
        "changed_sequences": sum(item["changed_sequences"] for item in reports),
    }
    write_json(output_root / "TOKEN_REPAIR_MANIFEST.json", manifest)
    seal = {
        "complete": True,
        "dataset_id": dataset_id,
        "producer_commit": commit,
        "manifest_sha256": sha256_file(output_root / "TOKEN_REPAIR_MANIFEST.json"),
        "sequences": manifest["sequences"],
        "tokens": manifest["tokens"],
        "fresh_reserve_and_independent_acceptance_required": True,
    }
    write_json(output_root / "_SUCCESS.json", seal)
    return seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "prepared-root", "tokenizer", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--pipeline-commit", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    print(
        json.dumps(
            repair_root(
                args.source_root,
                args.prepared_root,
                args.tokenizer,
                args.output_root,
                args.dataset_id,
                args.producer_commit,
                args.pipeline_commit,
                args.workers,
            )
        )
    )


if __name__ == "__main__":
    main()
