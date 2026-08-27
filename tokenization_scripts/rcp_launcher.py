"""Generic Run:ai launcher and sealer for configured tokenization jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline_pretrain.pipeline.tokens import read_token_map

SCHEMA_VERSION = "tokenizer-rcp-launch/v1"
SUCCESS_VERSION = "tokenizer-rcp-success/v1"
INDEX_HEADER = b"MMIDIDX\x00\x00"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,52}$")
NODE_POOL_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:G|Gi)$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to replace immutable file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _on_rcp(value: str, label: str) -> Path:
    path = Path(value)
    mount = Path("/mloscratch")
    if not path.is_absolute() or (path != mount and mount not in path.parents):
        raise ValueError(f"{label} must be an absolute /mloscratch path")
    return path


@dataclass(frozen=True)
class RcpTokenizerLaunchConfig:
    schema_version: str
    implementation_commit: str
    code_root: str
    python_bin: str
    dataset_config: str
    dataset_config_sha256: str
    processed_success_sha256: str
    work_root: str
    job_prefix: str
    image: str
    project: str
    pvc: str
    node_pool: str
    run_as_uid: int
    run_as_gid: int
    prepare_cpu: int
    prepare_memory: str
    tokenize_cpu: int
    tokenize_memory: str
    backoff_limit: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RcpTokenizerLaunchConfig:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError(
                "tokenizer RCP config fields differ; "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def load(
        cls, path: str | Path, *, expected_sha256: str
    ) -> RcpTokenizerLaunchConfig:
        if not SHA256_RE.fullmatch(expected_sha256):
            raise ValueError("launch config digest must be SHA-256")
        if sha256_file(path) != expected_sha256:
            raise ValueError("launch config digest differs")
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("tokenizer RCP config must be a JSON object")
        return cls.from_dict(value)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported tokenizer RCP config schema")
        if not COMMIT_RE.fullmatch(self.implementation_commit):
            raise ValueError("implementation commit must be full and lowercase")
        for value, label in (
            (self.dataset_config_sha256, "dataset config"),
            (self.processed_success_sha256, "processed success"),
        ):
            if not SHA256_RE.fullmatch(value):
                raise ValueError(f"{label} digest must be SHA-256")
        if not IMAGE_RE.fullmatch(self.image):
            raise ValueError("RCP image must be digest-addressed")
        if not NAME_RE.fullmatch(self.job_prefix):
            raise ValueError("job prefix is not a safe Run:ai name")
        if not NODE_POOL_RE.fullmatch(self.node_pool):
            raise ValueError("RCP node pool is invalid")
        if not all(
            isinstance(value, str) and value for value in (self.project, self.pvc)
        ):
            raise ValueError("Run:ai project and PVC must be non-empty")
        for value, label in (
            (self.run_as_uid, "run_as_uid"),
            (self.run_as_gid, "run_as_gid"),
            (self.prepare_cpu, "prepare_cpu"),
            (self.tokenize_cpu, "tokenize_cpu"),
        ):
            _positive(value, label)
        for value, label in (
            (self.prepare_memory, "prepare_memory"),
            (self.tokenize_memory, "tokenize_memory"),
        ):
            if not MEMORY_RE.fullmatch(value):
                raise ValueError(f"{label} is invalid")
        if (
            isinstance(self.backoff_limit, bool)
            or not isinstance(self.backoff_limit, int)
            or self.backoff_limit < 0
        ):
            raise ValueError("backoff_limit must be non-negative")
        paths = {
            label: _on_rcp(getattr(self, label), label)
            for label in (
                "code_root",
                "python_bin",
                "dataset_config",
                "work_root",
            )
        }
        if paths["code_root"] not in paths["dataset_config"].parents:
            raise ValueError("dataset config must be inside the pinned checkout")
        if paths["code_root"] in paths["python_bin"].parents:
            raise ValueError("Python runtime must be staged outside the checkout")
        if paths["python_bin"].name != "python3":
            raise ValueError("python_bin must name the staged python3 executable")


def load_dataset_config(config: RcpTokenizerLaunchConfig) -> dict[str, str]:
    if not Path(config.dataset_config).is_file():
        raise FileNotFoundError(config.dataset_config)
    if sha256_file(config.dataset_config) != config.dataset_config_sha256:
        raise ValueError("dataset config digest differs")
    script_root = Path(config.code_root) / "tokenization_scripts"
    command = 'set -euo pipefail; cd "$1"; set -a; source "$2"; env -0'
    output = subprocess.run(
        ["bash", "-c", command, "load-config", str(script_root), config.dataset_config],
        check=True,
        capture_output=True,
    ).stdout
    environment: dict[str, str] = {}
    for item in output.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode()] = value.decode()
    required = (
        "TOKENIZER",
        "TOKENIZER_NAME",
        "DATASET_NAME",
        "COLUMN_KEY",
        "ID_COLUMN",
        "PATH_TO_RAW_DATASET",
        "PATH_TO_OUTPUT_FOLDER",
        "PATH_TO_PREPROCESSING_METADATA",
        "DATASET_OUTPUT_FOLDER_NAME",
        "REQUIRED_DATASET_MARKER",
        "DATASET_MANIFEST",
        "DUMP_GROUP_FIELDS",
        "EXPECTED_GROUP_COUNT",
        "EXPECTED_GROUP_HEADS",
        "MAX_SEQUENCE_TOKENS",
        "NUMBER_OF_DATATROVE_TASKS",
    )
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise ValueError(f"dataset config is missing required values: {missing}")
    for key in (
        "PATH_TO_RAW_DATASET",
        "PATH_TO_OUTPUT_FOLDER",
        "PATH_TO_PREPROCESSING_METADATA",
        "DATASET_OUTPUT_FOLDER_NAME",
        "REQUIRED_DATASET_MARKER",
        "DATASET_MANIFEST",
    ):
        _on_rcp(environment[key], key)
    tokenizer = Path(environment["TOKENIZER"])
    if not tokenizer.is_absolute():
        tokenizer = (script_root / tokenizer).resolve()
    environment["TOKENIZER"] = str(tokenizer)
    if int(environment["MAX_SEQUENCE_TOKENS"]) < 1:
        raise ValueError("configured sequence limit must be positive")
    if int(environment["NUMBER_OF_DATATROVE_TASKS"]) < 1:
        raise ValueError("configured Datatrove task count must be positive")
    if environment["DUMP_GROUP_FIELDS"] != "category":
        raise ValueError("RCP grouped tokenization expects category-only grouping")
    return environment


def _checkout(config: RcpTokenizerLaunchConfig) -> str:
    head = subprocess.run(
        ["git", "-C", config.code_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != config.implementation_commit:
        raise ValueError("staged tokenizer checkout differs from its pin")
    dirty = subprocess.run(
        ["git", "-C", config.code_root, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError("staged tokenizer checkout is not clean")
    return head


def preflight(config: RcpTokenizerLaunchConfig) -> dict[str, Any]:
    values = load_dataset_config(config)
    if not Path(config.python_bin).is_file():
        raise FileNotFoundError(config.python_bin)
    marker = Path(values["REQUIRED_DATASET_MARKER"])
    if not marker.is_file() or sha256_file(marker) != config.processed_success_sha256:
        raise ValueError("processed dataset marker is absent or differs from its pin")
    for key in ("DATASET_MANIFEST", "TOKENIZER"):
        if not Path(values[key]).is_file():
            raise FileNotFoundError(values[key])
    head = _checkout(config)
    return {
        "schema_version": "tokenizer-rcp-preflight/v1",
        "implementation_commit": head,
        "dataset_config_sha256": config.dataset_config_sha256,
        "processed_success_sha256": config.processed_success_sha256,
        "tokenizer_sha256": sha256_file(values["TOKENIZER"]),
        "dataset": values["DATASET_NAME"],
        "max_sequence_tokens": int(values["MAX_SEQUENCE_TOKENS"]),
        "output_root": values["DATASET_OUTPUT_FOLDER_NAME"],
    }


def _base_environment(config: RcpTokenizerLaunchConfig) -> list[str]:
    return [
        f"export PYTHONPATH={shlex.quote(config.code_root)}",
        f"export PATH={shlex.quote(str(Path(config.python_bin).parent))}:$PATH",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1",
        "export TOKENIZATION_LAUNCH_BACKEND=rcp",
    ]


def _runtime_pin_checks(config: RcpTokenizerLaunchConfig) -> list[str]:
    """Return checks every payload repeats immediately before execution."""

    marker = (
        '"$REQUIRED_DATASET_MARKER"'  # populated by sourcing the pinned dataset config
    )
    return [
        f'test "$(git -C {shlex.quote(config.code_root)} rev-parse HEAD)" = {shlex.quote(config.implementation_commit)}',
        f'test -z "$(git -C {shlex.quote(config.code_root)} status --porcelain)"',
        f"test \"$(sha256sum {shlex.quote(config.dataset_config)} | cut -d' ' -f1)\" = {shlex.quote(config.dataset_config_sha256)}",
        f"source {shlex.quote(config.dataset_config)}",
        f"test \"$(sha256sum {marker} | cut -d' ' -f1)\" = {shlex.quote(config.processed_success_sha256)}",
    ]


def prepare_command(config: RcpTokenizerLaunchConfig) -> str:
    script = Path(config.code_root) / "tokenization_scripts" / "tokenize_script.sh"
    temp = Path(config.work_root) / "prepare-tmp"
    statements = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(temp))}",
        f"export TMPDIR={shlex.quote(str(temp))}",
        *_base_environment(config),
        f"cd {shlex.quote(str(Path(config.code_root) / 'tokenization_scripts'))}",
        *_runtime_pin_checks(config),
        (
            f"exec bash {shlex.quote(str(script))} "
            f"{shlex.quote(config.dataset_config)} --prepare-only"
        ),
    ]
    return "; ".join(statements)


def pending_dumps(
    config: RcpTokenizerLaunchConfig, values: dict[str, str]
) -> list[tuple[Path, Path, Path, Path]]:
    metadata = Path(values["PATH_TO_PREPROCESSING_METADATA"])
    dump_root = metadata / "dumps"
    if not dump_root.is_dir():
        raise FileNotFoundError("configured dumps are not prepared")
    result = []
    for path in sorted(dump_root.rglob("paths_file_*.txt")):
        relative = path.relative_to(dump_root)
        group = relative.parent
        dump = path.stem.removeprefix("paths_file_")
        output = Path(values["DATASET_OUTPUT_FOLDER_NAME"]) / group / f"dump-{dump}"
        logging = (
            Path(values["PATH_TO_OUTPUT_FOLDER"])
            / "logs/datatrove_logs"
            / values["TOKENIZER_NAME"]
            / values["DATASET_NAME"]
            / group
            / f"dump-{dump}"
        )
        completed = metadata / "completed-dumps" / group
        result.append((path, output, logging, completed))
    return result


def tokenize_command(
    config: RcpTokenizerLaunchConfig,
    paths_file: Path,
    output: Path,
    logging: Path,
    completed: Path,
) -> str:
    script = Path(config.code_root) / "tokenization_scripts" / "tokenize.sh"
    temporary = (
        Path(config.work_root)
        / "tokenize-tmp"
        / hashlib.sha256(str(paths_file).encode()).hexdigest()[:16]
    )
    args = [
        "bash",
        str(script),
        config.dataset_config,
        str(output),
        str(logging),
        str(paths_file),
        str(completed),
    ]
    statements = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(str(temporary))}",
        f"export TMPDIR={shlex.quote(str(temporary))}",
        *_base_environment(config),
        f"cd {shlex.quote(str(Path(config.code_root) / 'tokenization_scripts'))}",
        *_runtime_pin_checks(config),
        f"exec {shlex.join(args)}",
    ]
    return "; ".join(statements)


def runai_command(
    config: RcpTokenizerLaunchConfig,
    *,
    name: str,
    cpu: int,
    memory: str,
    payload: str,
) -> list[str]:
    return [
        "runai",
        "submit",
        "--name",
        name,
        "--project",
        config.project,
        "--image",
        config.image,
        "--gpu",
        "0",
        "--cpu",
        str(cpu),
        "--memory",
        memory,
        "--run-as-uid",
        str(config.run_as_uid),
        "--run-as-gid",
        str(config.run_as_gid),
        "--existing-pvc",
        f"claimname={config.pvc},path=/mloscratch",
        "--node-pools",
        config.node_pool,
        "--image-pull-policy",
        "Always",
        "--backoff-limit",
        str(config.backoff_limit),
        "--",
        "/bin/bash",
        "-lc",
        payload,
    ]


def _index(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) < 34 or payload[:9] != INDEX_HEADER:
        raise ValueError(f"invalid Megatron index header: {path}")
    version = struct.unpack_from("<Q", payload, 9)[0]
    dtype_code = payload[17]
    sequence_count = struct.unpack_from("<Q", payload, 18)[0]
    document_count = struct.unpack_from("<Q", payload, 26)[0]
    if version != 1 or dtype_code not in {4, 8}:
        raise ValueError(f"unsupported Megatron index: {path}")
    expected_bytes = 34 + 4 * sequence_count + 8 * sequence_count + 8 * document_count
    if len(payload) != expected_bytes:
        raise ValueError(f"Megatron index size differs: {path}")
    lengths = (
        struct.unpack_from(f"<{sequence_count}i", payload, 34) if sequence_count else ()
    )
    if any(length < 1 for length in lengths):
        raise ValueError(f"Megatron index has an empty sequence: {path}")
    return {
        "dtype_bytes": 4 if dtype_code == 4 else 2,
        "sequence_count": sequence_count,
        "document_count": document_count,
        "token_count": sum(lengths),
        "max_sequence_tokens": max(lengths, default=0),
    }


def validate_and_seal(
    config: RcpTokenizerLaunchConfig, values: dict[str, str]
) -> dict[str, Any]:
    output = Path(values["DATASET_OUTPUT_FOLDER_NAME"])
    success = output / "_SUCCESS.json"
    if success.exists():
        raise FileExistsError("refusing to reseal tokenized output")
    if list(
        (Path(values["PATH_TO_PREPROCESSING_METADATA"]) / "dumps").rglob(
            "paths_file_*.txt"
        )
    ):
        raise ValueError("tokenization has pending configured dumps")
    expected_heads = set(values["EXPECTED_GROUP_HEADS"].split(","))
    triples: dict[Path, set[str]] = {}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix == ".partial":
            raise ValueError(f"partial token output remains: {path}")
        if path.is_file() and path.suffix in {".bin", ".idx", ".map"}:
            triples.setdefault(path.with_suffix(""), set()).add(path.suffix)
    if not triples or any(
        suffixes != {".bin", ".idx", ".map"} for suffixes in triples.values()
    ):
        raise ValueError("tokenized output contains missing bin/idx/map triples")
    observed_heads: set[str] = set()
    counts = {"pairs": 0, "sequences": 0, "tokens": 0, "max_sequence_tokens": 0}
    manifest_rows = []
    for base in sorted(triples):
        relative_base = base.relative_to(output)
        if not relative_base.parts:
            raise ValueError("tokenized pair has no configured group")
        observed_heads.add(relative_base.parts[0])
        index_path = base.with_suffix(".idx")
        bin_path = base.with_suffix(".bin")
        map_path = base.with_suffix(".map")
        index = _index(index_path)
        if index["max_sequence_tokens"] > int(values["MAX_SEQUENCE_TOKENS"]):
            raise ValueError(
                f"tokenized sequence exceeds configured context: {index_path}"
            )
        if bin_path.stat().st_size != index["token_count"] * index["dtype_bytes"]:
            raise ValueError(f"token binary size differs from index: {bin_path}")
        token_map = read_token_map(map_path.read_bytes())
        if token_map["manifest"]["sequence_count"] != index["sequence_count"]:
            raise ValueError(f"token map count differs from index: {map_path}")
        if token_map["manifest"].get("index_sha256") != sha256_file(index_path):
            raise ValueError(f"token map does not bind the index: {map_path}")
        counts["pairs"] += 1
        counts["sequences"] += index["sequence_count"]
        counts["tokens"] += index["token_count"]
        counts["max_sequence_tokens"] = max(
            counts["max_sequence_tokens"], index["max_sequence_tokens"]
        )
        for path in (bin_path, index_path, map_path):
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(output).as_posix(),
                    "file_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if observed_heads != expected_heads:
        raise ValueError(
            f"tokenized category heads differ: {sorted(observed_heads)} != {sorted(expected_heads)}"
        )
    manifest_path = output / "manifest.jsonl"
    manifest_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows
    )
    if manifest_path.exists() and manifest_path.read_text() != manifest_payload:
        raise FileExistsError("refusing to replace a differing token manifest")
    _write_immutable(manifest_path, manifest_payload)
    summary = {
        "schema_version": SUCCESS_VERSION,
        "dataset": values["DATASET_NAME"],
        "implementation_commit": config.implementation_commit,
        "dataset_config": config.dataset_config,
        "dataset_config_sha256": config.dataset_config_sha256,
        "processed_success_sha256": config.processed_success_sha256,
        "prepared_examples_manifest_sha256": sha256_file(values["DATASET_MANIFEST"]),
        "tokenizer_sha256": sha256_file(values["TOKENIZER"]),
        "manifest_sha256": sha256_file(manifest_path),
        "categories": sorted(observed_heads),
        "counts": counts,
    }
    run_path = output / "TOKENIZATION_RUN.json"
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    _write_immutable(run_path, payload)
    _write_immutable(success, payload)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "render-prepare",
            "submit-prepare",
            "render",
            "submit",
            "validate",
        ),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RcpTokenizerLaunchConfig.load(
        args.config, expected_sha256=args.config_sha256
    )
    report = preflight(config)
    values = load_dataset_config(config)
    if args.action == "preflight":
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.action == "validate":
        print(json.dumps(validate_and_seal(config, values), sort_keys=True))
        return 0
    if args.action.endswith("prepare"):
        commands = [
            runai_command(
                config,
                name=f"{config.job_prefix}-prepare",
                cpu=config.prepare_cpu,
                memory=config.prepare_memory,
                payload=prepare_command(config),
            )
        ]
    else:
        commands = []
        for paths, output, logging, completed in pending_dumps(config, values):
            relative = paths.relative_to(
                Path(values["PATH_TO_PREPROCESSING_METADATA"]) / "dumps"
            )
            suffix = re.sub(r"[^a-z0-9-]+", "-", str(relative).lower()).strip("-")
            name = f"{config.job_prefix}-{hashlib.sha256(suffix.encode()).hexdigest()[:10]}"
            commands.append(
                runai_command(
                    config,
                    name=name,
                    cpu=config.tokenize_cpu,
                    memory=config.tokenize_memory,
                    payload=tokenize_command(config, paths, output, logging, completed),
                )
            )
    if args.action.startswith("render"):
        print(json.dumps(commands, indent=2))
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    print(json.dumps({**report, "submitted_jobs": len(commands)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
