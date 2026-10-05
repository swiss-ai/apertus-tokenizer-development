"""Check the sealed HPLT input contract and CPU worker argument forwarding."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(
    (ROOT / "tokenization_scripts/configs_apertus_v2").glob(
        "hplt-ia-finite-score-salvage-*.cfg"
    )
)
FIELDS = [
    "DATASET_NAME",
    "PATH_TO_RAW_DATASET",
    "TOKEN_MAP_SOURCE_ROOT",
    "DATASET_MANIFEST",
    "REQUIRED_DATASET_MARKER",
    "EXPECTED_GROUP_COUNT",
    "EXPECTED_GROUP_HEADS",
    "COLUMN_KEY",
    "ID_COLUMN",
    "GPUS",
    "CPUS_PER_TASK",
    "TOKENIZER_WORKERS",
    "TOKENIZER_THREADS",
    "TOKENIZER_BATCH_BYTES",
    "TOKENIZATION_VALIDATION_MODE",
    "PATH_TO_PREPROCESSING_METADATA",
    "DATASET_OUTPUT_FOLDER_NAME",
]


def config_values(path):
    command = 'source "$1"; shift; for key in "$@"; do printf "%s\\n" "${!key}"; done'
    result = subprocess.check_output(
        ["bash", "-c", command, "config", str(path), *FIELDS], text=True
    )
    return dict(zip(FIELDS, result.splitlines()))


@pytest.mark.parametrize("config", CONFIGS)
def test_manifest_identity_and_thread_budget(config):
    row = config_values(config)
    assert "finite-score-salvage-v1-20261005" in row["PATH_TO_RAW_DATASET"]
    assert row["TOKEN_MAP_SOURCE_ROOT"] == row["PATH_TO_RAW_DATASET"]
    assert (
        row["DATASET_MANIFEST"]
        == row["PATH_TO_RAW_DATASET"] + "/examples_manifest.jsonl"
    )
    assert (
        row["REQUIRED_DATASET_MARKER"] == row["PATH_TO_RAW_DATASET"] + "/_SUCCESS.json"
    )
    assert (row["COLUMN_KEY"], row["ID_COLUMN"]) == ("text", "id")
    assert row["TOKENIZATION_VALIDATION_MODE"] == "strict"
    assert row["GPUS"] == "0"
    assert int(row["TOKENIZER_WORKERS"]) * int(row["TOKENIZER_THREADS"]) <= int(
        row["CPUS_PER_TASK"]
    )
    assert row["TOKENIZER_BATCH_BYTES"] == "33554432"
    assert row["EXPECTED_GROUP_COUNT"] == "8"
    assert len(row["EXPECTED_GROUP_HEADS"].split(",")) == 8


def test_variants_have_disjoint_input_and_output_roots():
    assert len(CONFIGS) == 4
    rows = [config_values(config) for config in CONFIGS]
    assert len({r["PATH_TO_RAW_DATASET"] for r in rows}) == 4
    assert len({r["DATASET_OUTPUT_FOLDER_NAME"] for r in rows}) == 4


def test_worker_uses_absolute_paths_and_pinned_runtime(tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "git").write_text(
        "#!/bin/sh\necho fc1e5a86526e38c5f31aa6a099473f5a95b511e7\n"
    )
    (fakebin / "srun").write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "open(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n"
    )
    for path in fakebin.iterdir():
        path.chmod(0o755)
    capture = tmp_path / "capture.json"
    config = CONFIGS[0]
    row = config_values(config)
    paths_file = (
        row["PATH_TO_PREPROCESSING_METADATA"] + "/dumps/jpn_Jpan/paths_file_7.txt"
    )
    runtime = tmp_path / "runtime"
    edf = tmp_path / "runtime.toml"
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
    }
    subprocess.run(
        [
            "bash",
            str(ROOT / "tokenization_scripts/hplt_finite_score_worker.sbatch"),
            str(config),
            paths_file,
            str(runtime),
            str(edf),
        ],
        env=env,
        check=True,
    )
    args = json.loads(capture.read_text())
    assert "--gpus=0" in args and "--gres=none" in args
    assert f"--environment={edf}" in args
    assert f"PYTHONPATH={runtime}/src" in args
    assert "TOKENIZATION_LAUNCH_BACKEND=rcp" in args
    bash = args.index("bash")
    assert args[bash + 1] == str(ROOT / "tokenization_scripts/tokenize.sh")
    assert args[bash + 3] == row["DATASET_OUTPUT_FOLDER_NAME"] + "/jpn_Jpan/dump-7"
    assert args[bash + 5] == paths_file


@pytest.mark.parametrize("stage", ["all", "prepare", "finalize"])
def test_dataset_stage_reaches_allocated_container(tmp_path, stage):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "srun").write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "open(os.environ['CAPTURE'],'w').write(json.dumps(sys.argv[1:]))\n"
    )
    (fakebin / "srun").chmod(0o755)
    capture = tmp_path / "capture.json"
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
    }
    subprocess.run(
        [
            "bash",
            str(ROOT / "tokenization_scripts/hplt_finite_score_dataset.sbatch"),
            "config.cfg",
            "runtime",
            "edf.toml",
            "control",
            stage,
        ],
        env=env,
        check=True,
    )
    assert json.loads(capture.read_text())[-5:] == [
        "config.cfg",
        "runtime",
        "edf.toml",
        "control",
        stage,
    ]


def test_invalid_dataset_stage_fails_before_launch():
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "tokenization_scripts/hplt_finite_score_dataset.sbatch"),
            "config.cfg",
            "runtime",
            "edf.toml",
            "control",
            "unknown",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Stage must be" in result.stderr
