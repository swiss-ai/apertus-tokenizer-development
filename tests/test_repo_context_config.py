from pathlib import Path

ROOT = Path(__file__).parents[1]
CONFIG_NAME = "stackv31-repo-context-4k-v1.cfg"
CLARIDEN_CONFIG = ROOT / "tokenization_scripts/configs_apertus_v2" / CONFIG_NAME
RCP_CONFIG = ROOT / "tokenization_scripts/configs_apertus_v2_rcp" / CONFIG_NAME


def _assignments(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }


def test_repo_context_configs_use_only_sealed_prepared_examples():
    required = {
        "COLUMN_KEY=text",
        "ID_COLUMN=id",
        "REQUIRED_DATASET_MARKER=$PATH_TO_RAW_DATASET/_SUCCESS",
        "DATASET_MANIFEST=$PATH_TO_RAW_DATASET/examples_manifest.jsonl",
        "DUMP_GROUP_FIELDS=category",
        "EXPECTED_GROUP_COUNT=4",
        "EXPECTED_GROUP_HEADS=programming,markup,data,prose",
        "MAX_SEQUENCE_TOKENS=4096",
    }
    for path in (CLARIDEN_CONFIG, RCP_CONFIG):
        config = path.read_text(encoding="utf-8")
        assert required <= set(config.splitlines())
        assert "INCLUDE_BOOLEAN_COLUMN" not in config
        assert "lineage" not in config
        assert "stackv31-languages-v1" not in config


def test_repo_context_configs_separate_clariden_and_rcp_paths():
    clariden = _assignments(CLARIDEN_CONFIG)
    rcp = _assignments(RCP_CONFIG)

    differing_keys = {
        key for key in clariden | rcp if clariden.get(key) != rcp.get(key)
    }
    assert differing_keys == {
        "EXECUTION_SITE",
        "PATH_TO_RAW_DATASET",
        "PATH_TO_OUTPUT_FOLDER",
    }
    assert clariden["EXECUTION_SITE"] == "CLARIDEN"
    assert clariden["PATH_TO_RAW_DATASET"] == (
        "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/code/"
        "stackv31-repo-context-4k-v1"
    )
    assert clariden["PATH_TO_OUTPUT_FOLDER"] == (
        "/capstor/store/cscs/swissai/infra01/datasets_tokenized/"
        "stackv31-repo-context-4k-v1_apertus_v2"
    )
    assert rcp["EXECUTION_SITE"] == "RCP"
    assert rcp["PATH_TO_RAW_DATASET"] == (
        "/mloscratch/stackv31-repo-context-4k-v1"
    )
    assert rcp["PATH_TO_OUTPUT_FOLDER"] == (
        "/mloscratch/stackv31-repo-context-4k-v1_apertus_v2"
    )
