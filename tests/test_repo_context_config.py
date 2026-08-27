from pathlib import Path


def test_repo_context_config_uses_only_sealed_prepared_examples():
    config = (
        Path(__file__).parents[1]
        / "tokenization_scripts/configs_apertus_v2/stackv31-repo-context-4k-v1.cfg"
    ).read_text(encoding="utf-8")
    required = {
        "EXECUTION_SITE=RCP",
        "COLUMN_KEY=text",
        "ID_COLUMN=id",
        "REQUIRED_DATASET_MARKER=$PATH_TO_RAW_DATASET/_SUCCESS",
        "DATASET_MANIFEST=$PATH_TO_RAW_DATASET/examples_manifest.jsonl",
        "DUMP_GROUP_FIELDS=category",
        "EXPECTED_GROUP_COUNT=4",
        "EXPECTED_GROUP_HEADS=programming,markup,data,prose",
        "MAX_SEQUENCE_TOKENS=4096",
    }
    assert required <= set(config.splitlines())
    assert "INCLUDE_BOOLEAN_COLUMN" not in config
    assert "lineage" not in config
    assert "stackv31-languages-v1" not in config
