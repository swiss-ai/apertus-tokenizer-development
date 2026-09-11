import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
CLARIDEN_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2"
RCP_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2_rcp"
PROCESSING_COMMIT = "0000000000000000000000000000000000000000"
PROCESSING_ROOT = (
    f"/mloscratch/apertus-stem-data/runs/stem-processing-{PROCESSING_COMMIT}-r1"
)
TOKENIZATION_ROOT = (
    f"/mloscratch/apertus-stem-data/runs/stem-tokenization-{PROCESSING_COMMIT}-r1"
)
CLARIDEN_PROCESSED_ROOT = "/capstor/store/cscs/swissai/infra01/datasets"
CLARIDEN_TOKENIZED_ROOT = "/capstor/store/cscs/swissai/infra01/datasets_tokenized"
RELEASES = {
    "biocorpus-upstream-text-v1": "2:00:00",
    "superior-reasoning-apertus-inner-v1": "2:00:00",
    "synthetic-1-unverified-apertus-inner-v1": "2:00:00",
    "synthetic-1-verified-apertus-inner-v1": "2:00:00",
    "thebiocollection-free-text-upstream-text-v1": "5:00:00",
    "thebiocollection-instruction-upstream-text-v1": "3:00:00",
}
# Clariden `sinfo` partition time limits, measured 2026-09-11:
#     debug    1:30:00      normal*  12:00:00
#     low      1-00:00:00   xfer     1-00:00:00
# Reservation SD-69241-apertus-1-5-0 is PartitionName=normal and inherits the
# same cap, so any TIME above 12:00:00 on `normal` is rejected at submit.
NORMAL_PARTITION_TIME_LIMIT_SECONDS = 12 * 3600


def _assignments(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }


def test_stem_dataset_configs_are_paired_across_both_directories():
    assert len(RELEASES) == 6
    clariden = {
        path.stem for path in CLARIDEN_CONFIGS.glob("*.cfg") if path.stem in RELEASES
    }
    rcp = {path.stem for path in RCP_CONFIGS.glob("*.cfg") if path.stem in RELEASES}
    assert clariden == rcp == set(RELEASES)


def test_stem_dataset_configs_differ_only_in_physical_paths():
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
        assert clariden["TOKENIZER"] == rcp["TOKENIZER"]
        assert clariden["TOKENIZER_NAME"] == rcp["TOKENIZER_NAME"]

        clariden_processed = f"{CLARIDEN_PROCESSED_ROOT}/{release}"
        clariden_tokens = f"{CLARIDEN_TOKENIZED_ROOT}/{release}_apertus_v2"
        rcp_processed = f"{PROCESSING_ROOT}/{release}/processed"
        rcp_tokens = f"{TOKENIZATION_ROOT}/{release}_apertus_v2"

        assert clariden["PATH_TO_RAW_DATASET"] == clariden_processed
        assert clariden["PATH_TO_OUTPUT_FOLDER"] == clariden_tokens
        assert clariden["PATH_TO_PREPROCESSING_METADATA"] == "$PATH_TO_OUTPUT_FOLDER"
        assert rcp["PATH_TO_RAW_DATASET"] == rcp_processed
        assert rcp["PATH_TO_OUTPUT_FOLDER"] == rcp_tokens
        assert rcp["PATH_TO_PREPROCESSING_METADATA"] == "$PATH_TO_OUTPUT_FOLDER"


def test_stem_dataset_configs_keep_the_fused_provenance_key():
    for release in RELEASES:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            config = _assignments(directory / f"{release}.cfg")
            assert config["ID_COLUMN"] == "source_key"


def test_stem_dataset_token_maps_record_the_final_clariden_path():
    for release in RELEASES:
        clariden = _assignments(CLARIDEN_CONFIGS / f"{release}.cfg")
        rcp = _assignments(RCP_CONFIGS / f"{release}.cfg")
        expected = f"{CLARIDEN_PROCESSED_ROOT}/{release}"

        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == rcp["TOKEN_MAP_SOURCE_ROOT"]
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == expected
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == clariden["PATH_TO_RAW_DATASET"]


def test_stem_dataset_configs_do_not_encode_an_execution_site_flag():
    for release in RELEASES:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            config = _assignments(directory / f"{release}.cfg")
            assert "EXECUTION_SITE" not in config
            assert "TOKENIZATION_LAUNCH_BACKEND" not in config


def _wall_time_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def test_stem_dataset_wall_times_are_sized_for_the_measured_corpus():
    for release, expected_time in RELEASES.items():
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            config = _assignments(directory / f"{release}.cfg")
            assert config["TIME"] == expected_time
            assert config["DUMPS_NUMBER"] == "4096"


def test_stem_dataset_wall_times_fit_the_normal_partition_limit():
    for release in RELEASES:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            config = _assignments(directory / f"{release}.cfg")
            if config["PARTITION"] != "normal":
                continue
            assert (
                _wall_time_seconds(config["TIME"])
                <= NORMAL_PARTITION_TIME_LIMIT_SECONDS
            )


def test_stem_dataset_configs_are_valid_shell():
    for release in RELEASES:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            path = directory / f"{release}.cfg"
            result = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr
