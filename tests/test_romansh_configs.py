"""Guard release identities, provenance and RCP/Clariden configuration agreement."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "monolingual": "monolingual",
    "poly-aligned": "poly_aligned",
    "poly-non-aligned": "poly_non_aligned",
    "synthetic": "synthetic",
}
FIELDS = [
    "DATASET_NAME",
    "COLUMN_KEY",
    "ID_COLUMN",
    "PATH_TO_RAW_DATASET",
    "TOKEN_MAP_SOURCE_ROOT",
    "DATASET_OUTPUT_FOLDER_NAME",
    "DATASET_MANIFEST",
    "REQUIRED_DATASET_MARKER",
    "DUMP_GROUP_FIELDS",
    "EXPECTED_GROUP_HEADS",
    "MAX_SEQUENCE_TOKENS",
    "TOKENIZER_NAME",
    "EXTENSION",
    "REHYDRATE_FLAG",
    "TOKENIZATION_VALIDATION_MODE",
    "TOKENIZATION_LAUNCH_BACKEND",
]


def load_config(environment, name):
    path = ROOT / "tokenization_scripts" / environment / f"romansh-{name}.cfg"
    script = 'source "$1"; shift; for key in "$@"; do printf "%s\\n" "${!key-}"; done'
    values = subprocess.check_output(
        ["bash", "-c", script, "config", str(path), *FIELDS], text=True
    )
    return dict(zip(FIELDS, values.splitlines()))


@pytest.mark.parametrize("name,category", CONFIGS.items())
def test_configs_preserve_group_identity_and_canonical_map_root(name, category):
    clariden = load_config("configs_apertus_v2", name)
    rcp = load_config("configs_apertus_v2_rcp", name)
    assert rcp.pop("TOKENIZATION_LAUNCH_BACKEND") == "rcp"
    assert clariden.pop("TOKENIZATION_LAUNCH_BACKEND") == ""
    assert rcp == clariden
    assert clariden["PATH_TO_RAW_DATASET"].endswith(
        f"apertus-pretrain-romansh-v1/{category}"
    )
    assert clariden["TOKEN_MAP_SOURCE_ROOT"] == clariden["PATH_TO_RAW_DATASET"]
    assert clariden["ID_COLUMN"] == "source_key"
    assert clariden["EXPECTED_GROUP_HEADS"] == category
    assert clariden["DUMP_GROUP_FIELDS"] == "category"
    assert clariden["TOKENIZATION_VALIDATION_MODE"] == "strict"
    assert clariden["DATASET_MANIFEST"].endswith("/examples_manifest.jsonl")
    assert clariden["REQUIRED_DATASET_MARKER"].endswith("/_SUCCESS.json")
    assert clariden["DATASET_OUTPUT_FOLDER_NAME"].endswith(
        f"/preliminary_mul_200k/romansh-{name}"
    )


def test_output_and_inventory_roots_do_not_overlap():
    configs = [load_config("configs_apertus_v2_rcp", name) for name in CONFIGS]
    for field in [
        "PATH_TO_RAW_DATASET",
        "DATASET_OUTPUT_FOLDER_NAME",
        "DATASET_MANIFEST",
    ]:
        assert len({row[field] for row in configs}) == 4
