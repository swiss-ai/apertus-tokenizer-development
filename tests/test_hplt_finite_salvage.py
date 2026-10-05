"""Check the sealed HPLT input contract and CPU worker argument forwarding."""

import json
import os
import subprocess
import sys
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


@pytest.mark.parametrize("spooled", [False, True])
def test_worker_uses_absolute_paths_and_pinned_runtime(tmp_path, spooled):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "git").write_text(
        "#!/bin/sh\necho fc1e5a86526e38c5f31aa6a099473f5a95b511e7\n"
    )
    (fakebin / "python3").write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        "if sys.argv[1].endswith('hplt_finite_score_resume.py'): print(os.environ.get('DUMP_STATE','pending'))\n"
        "else: os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
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
    script = ROOT / "tokenization_scripts/hplt_finite_score_worker.sbatch"
    if spooled:
        copy = tmp_path / "slurm_script"
        copy.write_text(script.read_text())
        script = copy
    subprocess.run(
        [
            "bash",
            str(script),
            str(config),
            paths_file,
            str(runtime),
            str(edf),
        ],
        env=env,
        check=True,
    )
    args = json.loads(capture.read_text())
    assert "--nodes=1" in args
    assert "--gpus=0" in args and "--gres=none" in args
    assert "--cpus-per-task=128" in args
    assert "--exclusive" in args and "--exact" in args
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
        check=False,
    )
    assert result.returncode == 2
    assert "Stage must be" in result.stderr


@pytest.mark.parametrize("fail_one", [False, True])
@pytest.mark.parametrize("spooled", [False, True])
def test_two_dump_steps_receive_disjoint_cpu_requests_and_both_are_waited(
    tmp_path, fail_one, spooled
):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "git").write_text(
        "#!/bin/sh\necho fc1e5a86526e38c5f31aa6a099473f5a95b511e7\n"
    )
    (fakebin / "python3").write_text(
        f"#!{sys.executable}\nimport os,sys\n"
        "if sys.argv[1].endswith('hplt_finite_score_resume.py'): print(os.environ.get('DUMP_STATE','pending'))\n"
        "else: os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    (fakebin / "srun").write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "open(os.environ['CAPTURE']+'/'+str(os.getpid())+'.json','w').write(json.dumps(sys.argv[1:]))\n"
        "sys.exit(1 if os.environ['FAIL_ONE']=='1' and any(x.endswith('paths_file_1.txt') for x in sys.argv) else 0)\n"
    )
    for path in fakebin.iterdir():
        path.chmod(0o755)
    capture = tmp_path / "capture"
    capture.mkdir()
    config = CONFIGS[0]
    metadata = config_values(config)["PATH_TO_PREPROCESSING_METADATA"]
    paths = [metadata + f"/dumps/jpn_Jpan/paths_file_{i}.txt" for i in (0, 1)]
    env = {
        **os.environ,
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
        "FAIL_ONE": "1" if fail_one else "0",
        "SLURM_CPUS_PER_TASK": "256",
        "SLURM_MEM_PER_NODE": "524288",
    }
    script = ROOT / "tokenization_scripts/hplt_finite_score_dump_pair.sbatch"
    if spooled:
        copy = tmp_path / "slurm_script"
        copy.write_text(script.read_text())
        script = copy
    command = [
        "bash",
        str(script),
        str(config),
        "runtime",
        "edf",
        *paths,
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == int(fail_one)
    calls = [json.loads(path.read_text()) for path in capture.glob("*.json")]
    assert len(calls) == 2
    assert all(
        "--cpus-per-task=128" in call
        and "--exact" in call
        and "--exclusive" in call
        and "--mem=262144M" in call
        for call in calls
    )
    assert all(sum(path in call for path in paths) == 1 for call in calls)
    # Reject duplicate work and insufficient CPU budget before invoking a worker.
    for bad_paths, cpus in (([paths[0], paths[0]], "256"), (paths, "128")):
        result = subprocess.run(
            command[:5] + bad_paths,
            env={**env, "SLURM_CPUS_PER_TASK": cpus},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert len(list(capture.glob("*.json"))) == 2


def test_completed_worker_returns_before_standard_dump_cleanup(tmp_path):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "git").write_text(
        "#!/bin/sh\necho fc1e5a86526e38c5f31aa6a099473f5a95b511e7\n"
    )
    (fakebin / "python3").write_text("#!/bin/sh\necho completed\n")
    (fakebin / "srun").write_text("#!/bin/sh\nexit 44\n")
    for path in fakebin.iterdir():
        path.chmod(0o755)
    metadata, output = tmp_path / "metadata", tmp_path / "tokens"
    dump = output / "jpn_Jpan/dump-0"
    dump.mkdir(parents=True)
    sentinel = dump / "tokens.bin"
    sentinel.write_bytes(b"preserve accepted bytes")
    config = tmp_path / "config.cfg"
    config.write_text(
        f"TOKENIZER=../preliminary_mul_200k/tokenizer.json\nPATH_TO_PREPROCESSING_METADATA={metadata}\nDATASET_OUTPUT_FOLDER_NAME={output}\nREQUIRED_DATASET_MARKER=marker\nDATASET_MANIFEST=manifest\nCPUS_PER_TASK=128\n"
    )
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"]}
    command = [
        "bash",
        str(ROOT / "tokenization_scripts/hplt_finite_score_worker.sbatch"),
        str(config),
        str(metadata / "dumps/jpn_Jpan/paths_file_0.txt"),
        "runtime",
        "edf",
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "Keeping already completed dump" in result.stdout
    assert sentinel.read_bytes() == b"preserve accepted bytes"
    sentinel.unlink()
    dump.rmdir()
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "output is missing" in result.stderr
