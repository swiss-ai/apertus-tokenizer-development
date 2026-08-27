import struct
from pathlib import Path
from unittest.mock import patch

import pytest

from tokenization_scripts.rcp_launcher import (
    INDEX_HEADER,
    SCHEMA_VERSION,
    RcpTokenizerLaunchConfig,
    _index,
    prepare_command,
    runai_command,
    tokenize_command,
    validate_and_seal,
)


def _value():
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation_commit": "a" * 40,
        "code_root": "/mloscratch/homes/user/apertus-tokenizer-development",
        "python_bin": "/mloscratch/runtime/bin/python3",
        "dataset_config": (
            "/mloscratch/homes/user/apertus-tokenizer-development/"
            "tokenization_scripts/configs_apertus_v2/dataset.cfg"
        ),
        "dataset_config_sha256": "b" * 64,
        "processed_success_sha256": "c" * 64,
        "work_root": "/mloscratch/tokenizer-work",
        "job_prefix": "stackv31-tok",
        "image": "registry.example/tokenizer@sha256:" + "d" * 64,
        "project": "mlo-user",
        "pvc": "mlo-scratch",
        "node_pool": "v100",
        "run_as_uid": 1234,
        "run_as_gid": 1234,
        "prepare_cpu": 4,
        "prepare_memory": "16Gi",
        "tokenize_cpu": 64,
        "tokenize_memory": "256Gi",
        "backoff_limit": 1,
    }


def _write_index(path: Path, lengths=(3, 4)):
    pointers = []
    total = 0
    for length in lengths:
        pointers.append(total * 2)
        total += length
    documents = tuple(range(len(lengths) + 1))
    payload = b"".join(
        [
            INDEX_HEADER,
            struct.pack("<Q", 1),
            struct.pack("<B", 8),
            struct.pack("<Q", len(lengths)),
            struct.pack("<Q", len(documents)),
            struct.pack(f"<{len(lengths)}i", *lengths),
            struct.pack(f"<{len(pointers)}q", *pointers),
            struct.pack(f"<{len(documents)}q", *documents),
        ]
    )
    path.write_bytes(payload)


def test_commands_execute_only_the_pinned_standard_config_entrypoints():
    config = RcpTokenizerLaunchConfig.from_dict(_value())
    prepare = prepare_command(config)
    assert "tokenize_script.sh" in prepare
    assert "--prepare-only" in prepare
    assert config.dataset_config_sha256 in prepare
    worker = tokenize_command(
        config,
        Path("/mloscratch/metadata/dumps/programming/paths_file_0.txt"),
        Path("/mloscratch/tokens/programming/dump-0"),
        Path("/mloscratch/logs/programming/dump-0"),
        Path("/mloscratch/metadata/completed-dumps/programming"),
    )
    assert "tokenize.sh" in worker
    assert config.dataset_config in worker
    command = runai_command(
        config,
        name="stackv31-tok-one",
        cpu=64,
        memory="256Gi",
        payload=worker,
    )
    assert command[:2] == ["runai", "submit"]
    assert config.image in command
    assert "--existing-pvc" in command
    assert f"claimname={config.pvc},path=/mloscratch" in command
    assert command[-1] == worker


def test_index_parser_reads_lengths_and_rejects_truncation(tmp_path: Path):
    path = tmp_path / "tokens.idx"
    _write_index(path)
    assert _index(path) == {
        "dtype_bytes": 2,
        "sequence_count": 2,
        "document_count": 3,
        "token_count": 7,
        "max_sequence_tokens": 4,
    }
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size differs"):
        _index(path)


def test_validator_requires_four_complete_bounded_triples(tmp_path: Path):
    output = tmp_path / "tokens"
    metadata = tmp_path / "metadata"
    (metadata / "dumps").mkdir(parents=True)
    values = {
        "DATASET_OUTPUT_FOLDER_NAME": str(output),
        "PATH_TO_PREPROCESSING_METADATA": str(metadata),
        "EXPECTED_GROUP_HEADS": "programming,markup,data,prose",
        "MAX_SEQUENCE_TOKENS": "4096",
        "DATASET_NAME": "fixture",
        "TOKENIZER": str(tmp_path / "tokenizer.json"),
        "DATASET_MANIFEST": str(tmp_path / "examples_manifest.jsonl"),
    }
    Path(values["TOKENIZER"]).write_text("tokenizer")
    Path(values["DATASET_MANIFEST"]).write_text("manifest\n")
    for category in ("programming", "markup", "data", "prose"):
        base = output / category / "dump-0" / "00000_tokens"
        base.parent.mkdir(parents=True)
        _write_index(base.with_suffix(".idx"), lengths=(3,))
        base.with_suffix(".bin").write_bytes(b"\x00" * 6)
        base.with_suffix(".map").write_bytes(b"map")

    config = RcpTokenizerLaunchConfig.from_dict(_value())

    def token_map(path):
        index_path = path.with_suffix(".idx")
        return {
            "manifest": {
                "sequence_count": 1,
                "index_sha256": __import__("hashlib")
                .sha256(index_path.read_bytes())
                .hexdigest(),
            }
        }

    with patch(
        "tokenization_scripts.rcp_launcher.read_token_map",
        side_effect=lambda payload: token_map(
            next(path for path in output.rglob("*.map") if path.read_bytes() == payload)
        ),
    ):
        summary = validate_and_seal(config, values)
    assert summary["counts"]["pairs"] == 4
    assert summary["counts"]["max_sequence_tokens"] == 3
    assert (output / "_SUCCESS.json").read_bytes() == (
        output / "TOKENIZATION_RUN.json"
    ).read_bytes()


def test_config_rejects_mutable_image():
    value = _value()
    value["image"] = "registry.example/tokenizer:latest"
    with pytest.raises(ValueError, match="digest-addressed"):
        RcpTokenizerLaunchConfig.from_dict(value)
