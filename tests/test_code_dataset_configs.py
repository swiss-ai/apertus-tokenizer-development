from pathlib import Path

ROOT = Path(__file__).parents[1]
CLARIDEN_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2"
RCP_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2_rcp"
PROCESSING_ROOT = (
    "/mloscratch/marin-hero-code-data/runs/"
    "code-processing-de8d06cb2673aed4a37eb26e9a5a6afaca325926-fused-r2"
)
TOKENIZATION_ROOT = (
    "/mloscratch/marin-hero-code-data/runs/"
    "code-tokenization-de8d06cb2673aed4a37eb26e9a5a6afaca325926-fused-r2"
)
RELEASES = (
    "agenttrove-marin-role-v1",
    "coderforge-preview-marin-tool-v1",
    "davinci-dev-ctx-marin-markdown-v1",
    "davinci-dev-env-marin-tool-v1",
    "nemotron-terminal-corpus-marin-role-v1",
    "open-swe-traces-v1-reasoning-tool-v1",
    "openswe-environment-markdown-v1",
    "openswe-glm47-role-tool-v1",
    "swe-hero-openhands-marin-tool-v1",
    "swe-rebench-openhands-marin-tool-v2",
    "swe-rebench-v2-contree-upstream-text-v1",
    "swe-zero-12m-trajectories-marin-role-v1",
)


def _assignments(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }


def test_code_dataset_configs_have_matching_clariden_and_rcp_variants():
    for release in RELEASES:
        clariden = _assignments(CLARIDEN_CONFIGS / f"{release}.cfg")
        rcp = _assignments(RCP_CONFIGS / f"{release}.cfg")

        differing_keys = {
            key for key in clariden | rcp if clariden.get(key) != rcp.get(key)
        }
        assert differing_keys == {
            "PATH_TO_RAW_DATASET",
            "PATH_TO_OUTPUT_FOLDER",
        }
        assert clariden["DATASET_NAME"] == release
        assert rcp["DATASET_NAME"] == release
        assert clariden["COLUMN_KEY"] == rcp["COLUMN_KEY"] == "text"
        assert clariden["ID_COLUMN"] == rcp["ID_COLUMN"] == "source_key"
        assert clariden["TOKENIZER"] == rcp["TOKENIZER"]
        assert clariden["TOKENIZER_NAME"] == rcp["TOKENIZER_NAME"]
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == rcp["TOKEN_MAP_SOURCE_ROOT"]

        clariden_processed = (
            f"/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/code/{release}"
        )
        clariden_tokens = (
            "/capstor/store/cscs/swissai/infra01/datasets_tokenized/"
            f"{release}_apertus_v2"
        )
        rcp_processed = f"{PROCESSING_ROOT}/{release}/processed"
        rcp_tokens = f"{TOKENIZATION_ROOT}/{release}_apertus_v2"

        assert clariden["PATH_TO_RAW_DATASET"] == clariden_processed
        assert clariden["PATH_TO_PREPROCESSING_METADATA"] == "$PATH_TO_OUTPUT_FOLDER"
        assert clariden["PATH_TO_OUTPUT_FOLDER"] == clariden_tokens
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == clariden_processed
        assert rcp["PATH_TO_RAW_DATASET"] == rcp_processed
        assert rcp["PATH_TO_PREPROCESSING_METADATA"] == "$PATH_TO_OUTPUT_FOLDER"
        assert rcp["PATH_TO_OUTPUT_FOLDER"] == rcp_tokens


def test_code_dataset_configs_do_not_encode_an_execution_site_flag():
    for release in RELEASES:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            config = _assignments(directory / f"{release}.cfg")
            assert "EXECUTION_SITE" not in config
            assert "TOKENIZATION_LAUNCH_BACKEND" not in config
