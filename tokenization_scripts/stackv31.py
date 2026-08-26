"""Plan, run, and validate grouped Stack v3.1 tokenization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

RUN_SCHEMA = "stackv31-grouped-tokenization-v1"
VALIDATION_SCHEMA = "stackv31-grouped-validation-v1"
RUN_MANIFEST = "STACKV31_RUN.json"
DOCUMENT_COST_BYTES = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def safe_relative(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {value!r}")
    return path.as_posix()


def load_categories(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("languages", payload)
    categories: dict[str, dict[str, str]] = {}
    if isinstance(rows, list):
        items = ((row.get("slug"), row) for row in rows)
    elif isinstance(rows, dict):
        items = rows.items()
    else:
        raise TypeError("category map must contain a language list or object")
    for slug, metadata in items:
        if not isinstance(slug, str) or not isinstance(metadata, dict):
            raise TypeError("category map contains an invalid language entry")
        safe_relative(slug, "language slug")
        if "/" in slug:
            raise ValueError(f"language slug contains a path separator: {slug!r}")
        category = metadata.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"language {slug!r} has no category")
        categories[slug] = {
            "category": category,
            "language": str(metadata.get("name") or metadata.get("language") or ""),
        }
    return categories


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"manifest line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError("group manifest is empty")
    return rows


def language_plan(
    input_root: Path,
    manifest_path: Path,
    categories_path: Path,
    output_layout: str,
) -> list[dict[str, Any]]:
    categories = load_categories(categories_path)
    languages: dict[str, dict[str, Any]] = {}
    for row in read_manifest(manifest_path):
        slug = row.get("language_slug")
        if slug not in categories:
            raise ValueError(f"manifest language {slug!r} is absent from the category map")
        relative_path = safe_relative(str(row.get("relative_path", "")), "source path")
        source = input_root / relative_path
        if not source.is_file():
            raise ValueError(f"manifest source does not exist: {source}")
        metadata = categories[slug]
        row_category = row.get("category")
        if row_category and row_category != metadata["category"]:
            raise ValueError(
                f"language {slug!r} has conflicting categories "
                f"{row_category!r} and {metadata['category']!r}"
            )
        counts = {
            key: int(row[key])
            for key in ("rows", "included_rows", "content_bytes", "included_content_bytes")
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"manifest has negative counts for {relative_path}")
        if counts["included_rows"] > counts["rows"]:
            raise ValueError(f"included rows exceed rows for {relative_path}")
        entry = languages.setdefault(
            slug,
            {
                "language_slug": slug,
                "language": str(row.get("language") or metadata["language"]),
                "category": metadata["category"],
                "files": [],
                "rows": 0,
                "expected_sequences": 0,
                "content_bytes": 0,
                "included_content_bytes": 0,
            },
        )
        entry["files"].append(relative_path)
        entry["rows"] += counts["rows"]
        entry["expected_sequences"] += counts["included_rows"]
        entry["content_bytes"] += counts["content_bytes"]
        entry["included_content_bytes"] += counts["included_content_bytes"]
    for entry in languages.values():
        entry["files"].sort()
        entry["tasks"] = len(entry["files"])
        entry["output_relative"] = safe_relative(
            output_layout.format(
                category=entry["category"],
                language_slug=entry["language_slug"],
            ),
            "output layout",
        )
    return [languages[slug] for slug in sorted(languages)]


def language_cost(entry: dict[str, Any]) -> int:
    return int(entry["included_content_bytes"]) + DOCUMENT_COST_BYTES * int(
        entry["expected_sequences"]
    )


def rank_slice(tasks: int, shard: int, shards: int) -> tuple[int, int]:
    if tasks < 1 or shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid task slice")
    base, extra = divmod(tasks, shards)
    offset = shard * base + min(shard, extra)
    return offset, base + (1 if shard < extra else 0)


def assignments(plan: list[dict[str, Any]], target_jobs: int) -> list[dict[str, Any]]:
    if target_jobs < 1:
        raise ValueError("target jobs must be positive")
    target_cost = max(1, sum(language_cost(entry) for entry in plan) // target_jobs)
    large: list[dict[str, Any]] = []
    small: list[dict[str, Any]] = []
    for entry in plan:
        cost = language_cost(entry)
        shard_count = min(entry["tasks"], max(1, (cost + target_cost - 1) // target_cost))
        if shard_count > 1 and cost >= 2 * target_cost:
            for shard in range(shard_count):
                offset, count = rank_slice(entry["tasks"], shard, shard_count)
                if count:
                    large.append(
                        {
                            "language_slug": entry["language_slug"],
                            "rank_offset": offset,
                            "local_tasks": count,
                        }
                    )
        else:
            small.append(entry)

    bundle_count = min(len(small), max(1, target_jobs - len(large))) if small else 0
    bundles: list[list[dict[str, Any]]] = [[] for _ in range(bundle_count)]
    weights = [0] * bundle_count
    for entry in sorted(small, key=lambda item: (-language_cost(item), item["language_slug"])):
        index = weights.index(min(weights))
        bundles[index].append(
            {
                "language_slug": entry["language_slug"],
                "rank_offset": 0,
                "local_tasks": entry["tasks"],
            }
        )
        weights[index] += language_cost(entry)

    grouped = [[piece] for piece in large] + [bundle for bundle in bundles if bundle]
    return [
        {"assignment": index, "pieces": pieces}
        for index, pieces in enumerate(grouped)
    ]


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    if not (input_root / "_SUCCESS").is_file():
        raise ValueError(f"input is not sealed: {input_root}")
    summary_path = input_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("input summary is not complete")
    decision = summary.get("decision", {})
    source = summary.get("source", {})
    expected = {
        "policy tag": (decision.get("policy_tag"), args.expected_policy_tag),
        "signals revision": (
            decision.get("signals_revision"),
            args.expected_signals_revision,
        ),
        "source revision": (source.get("revision"), args.expected_source_revision),
    }
    for label, (observed, wanted) in expected.items():
        if wanted and observed != wanted:
            raise ValueError(f"unexpected {label}: {observed!r}")
    tokenizer_sha256 = sha256_file(args.tokenizer_path)
    if args.expected_tokenizer_sha256 and tokenizer_sha256 != args.expected_tokenizer_sha256:
        raise ValueError(f"unexpected tokenizer SHA-256: {tokenizer_sha256}")

    plan = language_plan(
        input_root,
        args.group_manifest,
        args.category_map,
        args.output_layout,
    )
    observed_categories = sorted({entry["category"] for entry in plan})
    if args.expected_languages and len(plan) != args.expected_languages:
        raise ValueError(f"expected {args.expected_languages} languages, found {len(plan)}")
    if args.expected_categories:
        wanted_categories = sorted(args.expected_categories.split(","))
        if observed_categories != wanted_categories:
            raise ValueError(
                f"expected categories {wanted_categories}, found {observed_categories}"
            )

    paths_root = args.work_root / "paths"
    for entry in plan:
        atomic_write(
            paths_root / f"{entry['language_slug']}.txt",
            "".join(path + "\n" for path in entry["files"]),
        )
    work = assignments(plan, args.target_jobs)
    run = {
        "schema_version": RUN_SCHEMA,
        "implementation_commit": args.implementation_commit or git_head(),
        "input_root": str(input_root),
        "output_root": str(args.output_root),
        "work_root": str(args.work_root),
        "paths_root": str(paths_root),
        "text_column": args.column,
        "id_column": args.id_column,
        "output_layout": args.output_layout,
        "selection": {
            "include_boolean_column": args.include_boolean_column,
            "exclusion_reason_column": args.exclusion_reason_column,
            "policy_tag": decision.get("policy_tag"),
            "policy_sha256": decision.get("policy_sha256"),
            "signals_revision": decision.get("signals_revision"),
        },
        "artifact": {
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(args.group_manifest),
            "category_map_sha256": sha256_file(args.category_map),
            "source_revision": source.get("revision"),
        },
        "tokenizer": {
            "path": str(args.tokenizer_path),
            "sha256": tokenizer_sha256,
            "batch_size": args.tokenizer_batch_size,
        },
        "languages": plan,
        "assignments": work,
        "totals": {
            "languages": len(plan),
            "categories": observed_categories,
            "source_files": sum(entry["tasks"] for entry in plan),
            "rows": sum(entry["rows"] for entry in plan),
            "expected_sequences": sum(entry["expected_sequences"] for entry in plan),
            "assignments": len(work),
        },
    }
    manifest_path = args.work_root / RUN_MANIFEST
    payload = json.dumps(run, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"prepared run already exists with different inputs: {manifest_path}")
    atomic_write(manifest_path, payload)
    print(json.dumps(run["totals"], sort_keys=True))
    return run


def load_run(work_root: Path) -> tuple[dict[str, Any], str]:
    path = work_root / RUN_MANIFEST
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version") != RUN_SCHEMA:
        raise ValueError(f"unsupported run manifest: {path}")
    return run, sha256_file(path)


def run_assignment(args: argparse.Namespace, preprocess_main=None) -> None:
    run, run_sha256 = load_run(args.work_root)
    if not 0 <= args.assignment < len(run["assignments"]):
        raise ValueError("assignment is outside the prepared run")
    assignment = run["assignments"][args.assignment]
    languages = {entry["language_slug"]: entry for entry in run["languages"]}
    marker_root = args.work_root / "completed" / run_sha256
    if preprocess_main is None:
        try:
            from .preprocess_megatron import main as preprocess_main
        except ImportError:
            from preprocess_megatron import main as preprocess_main

    for piece_index, piece in enumerate(assignment["pieces"]):
        marker = marker_root / f"{args.assignment:04d}-{piece_index:04d}.json"
        if marker.is_file():
            continue
        language = languages[piece["language_slug"]]
        pipeline = {
            "dataset": "stackv31-languages-v1",
            "language": language["language"],
            "language_slug": language["language_slug"],
            "language_category": language["category"],
            "selection": {
                **run["selection"],
                "artifact_summary_sha256": run["artifact"]["summary_sha256"],
                "artifact_manifest_sha256": run["artifact"]["manifest_sha256"],
            },
            "implementation_commit": run["implementation_commit"],
        }
        workers = min(args.workers, piece["local_tasks"]) if args.workers > 0 else -1
        tokenizer_args = argparse.Namespace(
            tokenizer_name_or_path=run["tokenizer"]["path"],
            eos_token=None,
            output_folder=str(Path(run["output_root"]) / language["output_relative"]),
            logging_dir=str(
                args.work_root / "logs" / f"assignment-{args.assignment:04d}" / language["language_slug"]
            ),
            n_tasks=language["tasks"],
            n_workers=workers,
            local_tasks=piece["local_tasks"],
            rank_offset=piece["rank_offset"],
            dataset=run["input_root"],
            paths_file=str(Path(run["paths_root"]) / f"{language['language_slug']}.txt"),
            column=run["text_column"],
            rehydrate="False",
            extension=".parquet",
            include_boolean_column=run["selection"]["include_boolean_column"],
            exclusion_reason_column=run["selection"]["exclusion_reason_column"],
            provenance_pipeline_json=pipeline,
            tokenizer_batch_size=run["tokenizer"]["batch_size"],
        )
        preprocess_main(tokenizer_args)
        atomic_json(
            marker,
            {
                "run_sha256": run_sha256,
                "assignment": args.assignment,
                "piece": piece_index,
                **piece,
            },
        )
    atomic_json(
        marker_root / f"assignment-{args.assignment:04d}.json",
        {"run_sha256": run_sha256, "assignment": args.assignment},
    )


def parquet_row(path: Path, row: int, column: str) -> Any:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    offset = 0
    for group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        if row < offset + count:
            return parquet.read_row_group(group, columns=[column]).slice(
                row - offset, 1
            ).to_pylist()[0][column]
        offset += count
    raise ValueError(f"row {row} is outside {path}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    from mixture_sampler.provenance import read_token_map, validate_token_map

    run, run_sha256 = load_run(args.work_root)
    output_root = Path(run["output_root"])
    map_root = args.map_root or output_root
    problems: list[str] = []
    sequence_count = 0
    map_bytes = 0
    token_bytes = 0
    resolved_languages = 0
    if args.require_assignment_markers:
        for assignment in run["assignments"]:
            marker = (
                args.work_root
                / "completed"
                / run_sha256
                / f"assignment-{assignment['assignment']:04d}.json"
            )
            if not marker.is_file():
                problems.append(f"assignment {assignment['assignment']} is incomplete")

    for language in run["languages"]:
        relative = Path(language["output_relative"])
        maps = sorted((map_root / relative).glob("*.map"))
        if not maps:
            problems.append(f"{language['language_slug']}: no token maps")
            continue
        language_sequences = 0
        for map_path in maps:
            try:
                result = validate_token_map(map_path)
                token_map = read_token_map(map_path)
                index_path = output_root / relative / f"{map_path.stem}.idx"
                binary_path = output_root / relative / f"{map_path.stem}.bin"
                if index_path.stat().st_size != token_map.manifest["index_bytes"]:
                    raise ValueError("index byte count does not match token map")
                if sha256_file(index_path) != token_map.manifest["index_sha256"]:
                    raise ValueError("index digest does not match token map")
                facts = token_map.manifest.get("pipeline", {})
                if facts.get("language_slug") != language["language_slug"]:
                    raise ValueError("token map records another language")
                if facts.get("language_category") != language["category"]:
                    raise ValueError("token map records another language category")
                language_sequences += result["row_count"]
                map_bytes += result["file_bytes"]
                token_bytes += index_path.stat().st_size + binary_path.stat().st_size
            except (OSError, KeyError, ValueError) as exception:
                problems.append(f"{map_path}: {exception}")
        sequence_count += language_sequences
        if language_sequences != language["expected_sequences"]:
            problems.append(
                f"{language['language_slug']}: expected {language['expected_sequences']} "
                f"sequences, found {language_sequences}"
            )
        try:
            token_map = read_token_map(maps[0])
            if len(token_map):
                location = token_map.resolve(0)
                root = Path(location["raw_dataset_root"])
                value = parquet_row(
                    root / location["path"],
                    location["row"],
                    location["text_column"],
                )
                if not isinstance(value, str):
                    raise ValueError("resolved source text is not a string")
                resolved_languages += 1
        except (OSError, KeyError, ValueError) as exception:
            problems.append(f"{language['language_slug']}: cannot resolve source row: {exception}")

    if sequence_count != run["totals"]["expected_sequences"]:
        problems.append(
            f"expected {run['totals']['expected_sequences']} total sequences, found {sequence_count}"
        )
    overhead = map_bytes / token_bytes if token_bytes else 0.0
    if overhead > args.max_map_overhead:
        problems.append(
            f"map overhead {overhead:.6%} exceeds {args.max_map_overhead:.6%}"
        )
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "complete": not problems,
        "run_sha256": run_sha256,
        "problems": problems,
        "languages": len(run["languages"]),
        "resolved_languages": resolved_languages,
        "sequences": sequence_count,
        "map_bytes": map_bytes,
        "token_bytes": token_bytes,
        "map_overhead": overhead,
    }
    atomic_json(args.work_root / "STACKV31_VALIDATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(1)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--input-root", type=Path, required=True)
    prepare_parser.add_argument("--group-manifest", type=Path, required=True)
    prepare_parser.add_argument("--category-map", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--work-root", type=Path, required=True)
    prepare_parser.add_argument("--tokenizer-path", type=Path, required=True)
    prepare_parser.add_argument("--column", default="content")
    prepare_parser.add_argument("--id-column", default="content_id")
    prepare_parser.add_argument("--include-boolean-column", default="apertus_include")
    prepare_parser.add_argument("--exclusion-reason-column", default="exclusion_reason")
    prepare_parser.add_argument("--output-layout", default="{category}/{language_slug}")
    prepare_parser.add_argument("--expected-languages", type=int, default=0)
    prepare_parser.add_argument("--expected-categories", default="")
    prepare_parser.add_argument("--expected-policy-tag", default="")
    prepare_parser.add_argument("--expected-signals-revision", default="")
    prepare_parser.add_argument("--expected-source-revision", default="")
    prepare_parser.add_argument("--expected-tokenizer-sha256", default="")
    prepare_parser.add_argument("--implementation-commit", default="")
    prepare_parser.add_argument("--target-jobs", type=int, default=24)
    prepare_parser.add_argument("--tokenizer-batch-size", type=int, default=1000)

    run_parser = commands.add_parser("run-assignment")
    run_parser.add_argument("--work-root", type=Path, required=True)
    run_parser.add_argument("--assignment", type=int, required=True)
    run_parser.add_argument("--workers", type=int, default=16)

    count_parser = commands.add_parser("assignment-count")
    count_parser.add_argument("--work-root", type=Path, required=True)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--work-root", type=Path, required=True)
    validate_parser.add_argument("--map-root", type=Path, default=None)
    validate_parser.add_argument("--max-map-overhead", type=float, default=0.02)
    validate_parser.add_argument("--require-assignment-markers", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-assignment":
        run_assignment(args)
    elif args.command == "assignment-count":
        run, _ = load_run(args.work_root)
        print(len(run["assignments"]))
    else:
        validate(args)


if __name__ == "__main__":
    main()
