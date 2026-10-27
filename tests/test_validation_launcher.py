"""A disabled prepared-metadata guard must retain a positive payload bound."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tokenization_scripts/validate_tokenization.sh"
)


@pytest.mark.parametrize(
    "guard,override,expected", [(4096, None, "4096"), (0, 2147483647, "2147483647")]
)
def test_validation_bound_is_independent_of_prepared_metadata(
    tmp_path, guard, override, expected
):
    config = tmp_path / "input.cfg"
    config.write_text(
        "TOKENIZER=/unused/tokenizer.json\n"
        "TOKENIZER_NAME=test\nDATASET_NAME=test\n"
        "PATH_TO_RAW_DATASET=/unused/prepared\n"
        "PATH_TO_OUTPUT_FOLDER=/unused/tokens\n"
        "DATASET_MANIFEST=/unused/prepared/examples_manifest.jsonl\n"
        "REQUIRED_DATASET_MARKER=/unused/prepared/_SUCCESS.json\n"
        "COLUMN_KEY=text\nEXPECTED_GROUP_HEADS=monolingual\n"
        f"MAX_SEQUENCE_TOKENS={guard}\n"
        + (
            f"TOKENIZATION_VALIDATION_MAX_SEQUENCE_TOKENS={override}\n"
            if override is not None
            else ""
        )
    )
    capture = tmp_path / "arguments"
    executable = tmp_path / "python3"
    executable.write_text('#!/bin/sh\nprintf "%s\\0" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CAPTURE": str(capture),
    }
    subprocess.run(
        ["bash", str(SCRIPT), str(config), "producer-commit"],
        env=environment,
        check=True,
        capture_output=True,
    )
    arguments = capture.read_bytes().decode().rstrip("\0").split("\0")
    assert arguments[arguments.index("--max-sequence-tokens") + 1] == expected
    assert (
        arguments[arguments.index("--implementation-commit") + 1] == "producer-commit"
    )
