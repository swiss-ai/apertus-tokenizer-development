"""Qualify and seal HPLT kept-token unions without rewriting their parents."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from data_pipeline_pretrain.pipeline.tokens.megatron_index import read_megatron_index
from data_pipeline_pretrain.pipeline.tokens.megatron_tokenizer import (
    megatron_post_processor,
)
from data_pipeline_pretrain.pipeline.tokens.token_map import (
    read_token_map,
    resolve_token_map,
)
from tokenizers import Tokenizer

TOKENIZER_SHA256 = "cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9"
OLD_COUNTS = {
    "dclm-10": (175057450048, 17505289999, 157552160049),
    "dclm-33": (338157527288, 33815330069, 304342197219),
    "fwedu-10": (64434544310, 2249727246, 62184817064),
    "fwedu-33": (218837244050, 17709682945, 201127561105),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(partial, path)


def discover(root, reserved=False):
    root = Path(root)
    files = {suffix: set() for suffix in (".bin", ".idx", ".map")}
    for path in root.rglob("*"):
        if path.suffix not in files or not path.is_file():
            continue
        relative = path.relative_to(root)
        if not reserved and "_reserved" in relative.parts:
            continue
        files[path.suffix].add(relative.as_posix()[:-4])
    if files[".bin"] != files[".idx"] or files[".map"] - files[".idx"]:
        raise ValueError(f"incomplete or orphaned token pair in {root}")
    return sorted(files[".idx"])


def witness(prefix, index, token_map, tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.post_processor = megatron_post_processor()
    source = token_map["manifest"]
    if source.get("tokenizer", {}).get("sha256") != TOKENIZER_SHA256:
        raise ValueError("recovered map describes another tokenizer")
    binary = np.memmap(prefix.with_suffix(".bin"), mode="r", dtype="<u4")
    examples = []
    positions = sorted({0, index.sequence_count - 1, int(np.argmax(index.lengths))})
    for ordinal in positions:
        location = resolve_token_map(token_map, ordinal)
        path = Path(source["raw_dataset_root"]) / location["path"]
        file = pq.ParquetFile(path)
        row = location["row"]
        for group in range(file.num_row_groups):
            count = file.metadata.row_group(group).num_rows
            if row < count:
                record = (
                    file.read_row_group(
                        group, columns=["id", "text"], use_threads=False
                    )
                    .slice(row, 1)
                    .to_pylist()[0]
                )
                break
            row -= count
        else:
            raise ValueError("witness source row is missing")
        expected = np.asarray(tokenizer.encode(record["text"]).ids, dtype="<u4")
        start = int(index.pointers[ordinal]) // 4
        actual = binary[start : start + int(index.lengths[ordinal])]
        if not np.array_equal(expected, actual):
            raise ValueError(
                f"recovered tokens differ from source encoding: {prefix}/{ordinal}"
            )
        examples.append(
            {
                "sequence": ordinal,
                "id": record["id"],
                "source_file": str(path),
                "source_row": location["row"],
                "text_sha256": hashlib.sha256(record["text"].encode()).hexdigest(),
                "tokens": len(expected),
                "exact_reencoding": True,
            }
        )
    return examples


def qualify(task):
    root, relative, role, control, tokenizer_path = task
    prefix = Path(root) / relative
    paths = [
        prefix.with_suffix(suffix)
        for suffix in (".bin", ".idx", ".map")
        if prefix.with_suffix(suffix).is_file()
    ]
    before = {str(path): identity(path) for path in paths}
    if role == "recovered" and len(paths) != 3:
        raise ValueError("recovered token pair has no source map")
    recipe = sha(__file__)
    receipt_path = Path(control) / (
        hashlib.sha256(str(prefix).encode()).hexdigest() + ".json"
    )
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if (
            old["identities"] != before
            or old["producer_code_sha256"] != recipe
            or old["role"] != role
        ):
            raise ValueError("qualified token parent/recipe changed")
        return old
    index = read_megatron_index(
        prefix.with_suffix(".idx"), bin_size=prefix.with_suffix(".bin").stat().st_size
    )
    if index.dtype_code != 4 or index.sequence_count < 1:
        raise ValueError("expected nonempty uint32 token pair")
    artifacts = [
        {
            "parent": str(path),
            "relative_path": f"{role}/{relative}{path.suffix}",
            "sha256": sha(path),
            "bytes": before[str(path)][0],
        }
        for path in paths
    ]
    examples = []
    if prefix.with_suffix(".map").is_file():
        token_map = read_token_map(prefix.with_suffix(".map").read_bytes())
        manifest = token_map["manifest"]
        if (
            manifest["sequence_count"] != index.sequence_count
            or manifest["token_count"] != index.token_count
            or manifest["index_sha256"]
            != next(
                row["sha256"] for row in artifacts if row["parent"].endswith(".idx")
            )
        ):
            raise ValueError("token map/index disagreement")
        if role == "recovered":
            examples = witness(prefix, index, token_map, tokenizer_path)
    if before != {str(path): identity(path) for path in paths}:
        raise ValueError("token parent changed during qualification")
    result = {
        "parent_prefix": str(prefix),
        "relative_prefix": f"{role}/{relative}",
        "role": role,
        "tokens": index.token_count,
        "sequences": index.sequence_count,
        "identities": before,
        "artifacts": artifacts,
        "examples": examples,
        "producer_code_sha256": recipe,
    }
    atomic_json(receipt_path, result)
    return result


def reserve_counts(root):
    root = Path(root) / "_reserved"
    return sum(
        read_megatron_index(
            root / (prefix + ".idx"), bin_size=(root / (prefix + ".bin")).stat().st_size
        ).token_count
        for prefix in discover(root, reserved=True)
    )


def qualify_root(root, role, control, tokenizer, workers):
    prefixes = discover(root)
    if not prefixes:
        raise ValueError("empty kept-token parent")
    tasks = [
        (str(root), prefix, role, str(control), str(tokenizer)) for prefix in prefixes
    ]
    if workers == 1:
        results = [qualify(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            results = list(pool.map(qualify, tasks))
    if discover(root) != prefixes:
        raise ValueError("token parent population changed")
    return results


def seal_union(output, entries, accounting, pins, variant):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [artifact for entry in entries for artifact in entry["artifacts"]]
    expected = {row["relative_path"] for row in artifacts} | {
        "TOKEN_UNION_MANIFEST.jsonl",
        "TOKEN_UNION_MANIFEST.jsonl.partial",
        "_SUCCESS.json",
        "_SUCCESS.json.partial",
    }
    for path in output.rglob("*"):
        if (path.is_file() or path.is_symlink()) and path.relative_to(
            output
        ).as_posix() not in expected:
            raise ValueError("unexpected file in token union")
    for row in artifacts:
        target = output / row["relative_path"]
        parent = Path(row["parent"])
        if target.is_symlink() and target.resolve() == parent.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise ValueError("token union link differs from its parent")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(parent, target)
    for entry in entries:
        if entry["identities"] != {
            path: identity(path) for path in entry["identities"]
        }:
            raise ValueError("qualified parent changed before sealing")
    payload = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    manifest = output / "TOKEN_UNION_MANIFEST.jsonl"
    seal = output / "_SUCCESS.json"
    if seal.exists():
        old = json.loads(seal.read_text())
        if (
            manifest.read_text() != payload
            or old["accounting"] != accounting
            or old["pins"] != {**pins, "token_union_manifest_sha256": sha(manifest)}
        ):
            raise ValueError("sealed token union changed")
        return old
    partial = manifest.with_name(manifest.name + ".partial")
    partial.write_text(payload)
    os.replace(partial, manifest)
    result = {
        "complete": True,
        "smoke": False,
        "release": "hplt-ia-finite-score-salvage-kept-v1",
        "variant": variant,
        "output_root": str(output),
        "producer_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parents[1]),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "pins": {**pins, "token_union_manifest_sha256": sha(manifest)},
        "accounting": accounting,
        "pairs": len(entries),
        "limitation": "Provisional finite-score selection repair; finite embedding corruption remains eligible. Original token provenance availability is preserved, not retroactively reconstructed.",
    }
    atomic_json(seal, result)
    return result


def validate_recovered_parents(source, split):
    pins = source.get("pins", {})
    # Strict token validation requires a production prepared parent, but does
    # not repeat its smoke flag in the token seal.
    if (
        source.get("complete") is not True
        or source.get("smoke", False) is not False
        or source.get("constraints", {}).get("validation_mode") != "strict"
        or pins.get("tokenizer_sha256") != TOKENIZER_SHA256
        or any(
            not isinstance(pins.get(key), str)
            or re.fullmatch(pattern, pins[key]) is None
            for key, pattern in (
                ("prepared_marker_sha256", r"[0-9a-f]{64}"),
                ("prepared_examples_manifest_sha256", r"[0-9a-f]{64}"),
                ("implementation_commit", r"[0-9a-f]{40}"),
                ("validator_commit", r"[0-9a-f]{40}"),
            )
        )
        or split.get("policy") != "long-context-reserve-v4"
    ):
        raise ValueError("recovered source/split is not qualified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["original", "seal"])
    parser.add_argument("--variant", required=True, choices=OLD_COUNTS)
    parser.add_argument(
        "--original",
        required=True,
        type=Path,
        help="immutable original kept split root",
    )
    parser.add_argument(
        "--recovered", type=Path, help="sealed recovered .lc-reserve-v4 split root"
    )
    parser.add_argument("--output", type=Path, help="fresh union source root")
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="exact tokenizer.json used by the producer",
    )
    parser.add_argument(
        "--control",
        required=True,
        type=Path,
        help="persistent pair qualification receipts",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="concurrent read/hash processes (default: 16)",
    )
    args = parser.parse_args()
    if args.workers < 1 or (
        args.stage == "seal" and (args.recovered is None or args.output is None)
    ):
        parser.error("positive workers and seal recovered/output roots are required")
    if sha(args.tokenizer) != TOKENIZER_SHA256:
        raise ValueError("tokenizer differs from approved encoding")
    old = qualify_root(
        args.original, "original", args.control, args.tokenizer, args.workers
    )
    old_kept = sum(row["tokens"] for row in old)
    old_reserved = reserve_counts(args.original)
    if (old_kept + old_reserved, old_reserved, old_kept) != OLD_COUNTS[args.variant]:
        raise ValueError("original measured supply differs from the accepted ledger")
    if args.stage == "original":
        atomic_json(
            args.control / "original.json",
            {
                "variant": args.variant,
                "counts": OLD_COUNTS[args.variant],
                "pairs": len(old),
                "qualified": True,
            },
        )
        return
    source_marker = (
        args.recovered.with_name(args.recovered.name.removesuffix(".lc-reserve-v4"))
        / "_SUCCESS.json"
    )
    split_marker = args.recovered / "_SPLIT_SUCCESS.json"
    source, split = (
        json.loads(source_marker.read_text()),
        json.loads(split_marker.read_text()),
    )
    validate_recovered_parents(source, split)
    new = qualify_root(
        args.recovered, "recovered", args.control, args.tokenizer, args.workers
    )
    new_kept, new_reserved = (
        sum(row["tokens"] for row in new),
        reserve_counts(args.recovered),
    )
    if new_kept + new_reserved != source["totals"]["tokens"]:
        raise ValueError("recovered token conservation failed")
    accounting = {
        "original": {
            "source_tokens": old_kept + old_reserved,
            "reserved_tokens": old_reserved,
            "kept_tokens": old_kept,
        },
        "recovered": {
            "source_tokens": new_kept + new_reserved,
            "reserved_tokens": new_reserved,
            "kept_tokens": new_kept,
        },
        "combined": {
            "source_tokens": old_kept + old_reserved + new_kept + new_reserved,
            "reserved_tokens": old_reserved + new_reserved,
            "kept_tokens": old_kept + new_kept,
        },
    }
    pins = {
        "tokenizer_sha256": TOKENIZER_SHA256,
        "recovered_tokenization_marker_sha256": sha(source_marker),
        "recovered_split_marker_sha256": sha(split_marker),
    }
    print(
        json.dumps(
            seal_union(args.output, old + new, accounting, pins, args.variant),
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
