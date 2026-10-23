import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
CLARIDEN_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2"
RCP_CONFIGS = ROOT / "tokenization_scripts/configs_apertus_v2_rcp"
PROCESSING_COMMIT = "c0c24b40ec8581c31d70ebc1daa98f817fd2d6ab"
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
    "synthetic-1-unverified-apertus-inner-v2": "2:00:00",
    "synthetic-1-verified-apertus-inner-v2": "2:00:00",
    "thebiocollection-free-text-upstream-text-v1": "5:00:00",
    "thebiocollection-instruction-upstream-text-v1": "3:00:00",
}
# Clariden `sinfo` partition time limits, measured 2026-09-11:
#     debug    1:30:00      normal*  12:00:00
#     low      1-00:00:00   xfer     1-00:00:00
# Reservation SD-69241-apertus-1-5-0 is PartitionName=normal and inherits the
# same cap, so any TIME above 12:00:00 on `normal` is rejected at submit.
NORMAL_PARTITION_TIME_LIMIT_SECONDS = 12 * 3600

# The SYNTHETIC-1 v2 trees exist only on Clariden: they are the v1 run output with
# the licence-excluded collections removed (see DERIVATION.json in each tree), not
# the output of an RCP processing run, so they have no RCP counterpart.
CLARIDEN_ONLY = {
    "synthetic-1-unverified-apertus-inner-v2",
    "synthetic-1-verified-apertus-inner-v2",
}
# The unfiltered SYNTHETIC-1 v1 trees still carry the collections the licence
# review rejected and must not be tokenizable.
WITHDRAWN = {
    "synthetic-1-unverified-apertus-inner-v1",
    "synthetic-1-verified-apertus-inner-v1",
}

# Token counts of the tokenized trees on Clariden: the sum of the sequence lengths in
# each tree's sixteen `.idx` files under preliminary_mul_200k (vocabulary 200064,
# tokenizer.json sha256 cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9).
MEASURED_TOKENS = {
    "biocorpus-upstream-text-v1": 9_963_798_722,
    "superior-reasoning-apertus-inner-v1": 7_435_147_974,
    "synthetic-1-unverified-apertus-inner-v2": 1_788_357_966,
    "synthetic-1-verified-apertus-inner-v2": 1_903_905_109,
    "thebiocollection-free-text-upstream-text-v1": 38_939_153_636,
    "thebiocollection-instruction-upstream-text-v1": 21_187_485_660,
}
# Slowest 1x144 baseline rate implied by the PR #16 benchmark (Nemotron-V2:
# 5.48M tok/s candidate over a 6.45x speedup). The SYNTHETIC-1 v2 dump jobs ran at
# about 1M tok/s each, including start-up, so the rate is a conservative floor.
BASELINE_TOKENS_PER_SECOND = 0.850e6
# The producing pipelines write `part-${rank}.parquet` through a ParquetWriter
# with max_file_size 2 GiB, and TheBioCollection free-text alone is 32 upstream
# shards / 14.84 GiB compressed, so every processed tree has many parts.
# Sixteen dumps keep job-launch overhead bounded while preserving parallelism;
# four is still a pessimistic floor for the wall-time calculation.
ASSUMED_MINIMUM_DUMPS = 4
WALL_TIME_MARGIN = 1.5
# swe-rebench-v2-contree-upstream-text-v1 needed 1:00:00 raised to 3:00:00 under
# this exact config shape (commit ba736f7), so nothing here gets the bare default.
MINIMUM_WALL_TIME_HOURS = 2


def _assignments(path: Path) -> dict[str, str]:
    return {
        key: value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
        for key, value in [line.split("=", 1)]
    }


def _sites(release: str) -> tuple[Path, ...]:
    if release in CLARIDEN_ONLY:
        return (CLARIDEN_CONFIGS,)
    return (CLARIDEN_CONFIGS, RCP_CONFIGS)


def test_stem_dataset_configs_are_paired_across_both_directories():
    assert len(RELEASES) == 6
    clariden = {
        path.stem for path in CLARIDEN_CONFIGS.glob("*.cfg") if path.stem in RELEASES
    }
    rcp = {path.stem for path in RCP_CONFIGS.glob("*.cfg") if path.stem in RELEASES}
    assert clariden == set(RELEASES)
    assert rcp == set(RELEASES) - CLARIDEN_ONLY


def test_unfiltered_synthetic1_has_no_config():
    for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
        for release in WITHDRAWN:
            assert not (directory / f"{release}.cfg").exists()


def test_stem_dataset_configs_differ_only_in_physical_paths():
    for release in set(RELEASES) - CLARIDEN_ONLY:
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
        for directory in _sites(release):
            config = _assignments(directory / f"{release}.cfg")
            assert config["ID_COLUMN"] == "source_key"


def test_stem_dataset_token_maps_record_the_final_clariden_path():
    for release in RELEASES:
        expected = f"{CLARIDEN_PROCESSED_ROOT}/{release}"
        clariden = _assignments(CLARIDEN_CONFIGS / f"{release}.cfg")
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == expected
        assert clariden["TOKEN_MAP_SOURCE_ROOT"] == clariden["PATH_TO_RAW_DATASET"]
        assert clariden["PATH_TO_OUTPUT_FOLDER"] == (
            f"{CLARIDEN_TOKENIZED_ROOT}/{release}_apertus_v2"
        )
        for directory in _sites(release):
            config = _assignments(directory / f"{release}.cfg")
            assert config["TOKEN_MAP_SOURCE_ROOT"] == expected


def test_stem_dataset_configs_do_not_encode_an_execution_site_flag():
    for release in RELEASES:
        for directory in _sites(release):
            config = _assignments(directory / f"{release}.cfg")
            assert "EXECUTION_SITE" not in config
            assert "TOKENIZATION_LAUNCH_BACKEND" not in config


def _wall_time_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def test_stem_dataset_wall_times_are_sized_for_the_measured_corpus():
    for release, expected_time in RELEASES.items():
        for directory in _sites(release):
            config = _assignments(directory / f"{release}.cfg")
            assert config["TIME"] == expected_time
            assert config["DUMPS_NUMBER"] == "16"


def test_stem_dataset_wall_times_follow_the_documented_derivation():
    for release, expected_time in RELEASES.items():
        per_dump = MEASURED_TOKENS[release] / ASSUMED_MINIMUM_DUMPS
        hours = per_dump / BASELINE_TOKENS_PER_SECOND / 3600
        derived = max(MINIMUM_WALL_TIME_HOURS, math.ceil(hours * WALL_TIME_MARGIN))
        assert expected_time == f"{derived}:00:00"


def test_stem_dataset_wall_times_fit_the_normal_partition_limit():
    for release in RELEASES:
        for directory in _sites(release):
            config = _assignments(directory / f"{release}.cfg")
            if config["PARTITION"] != "normal":
                continue
            assert (
                _wall_time_seconds(config["TIME"])
                <= NORMAL_PARTITION_TIME_LIMIT_SECONDS
            )


def test_stem_dataset_configs_are_valid_shell():
    for release in RELEASES:
        for directory in _sites(release):
            path = directory / f"{release}.cfg"
            result = subprocess.run(
                ["bash", "-n", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, result.stderr


CORRECTED_SYNTHETIC1 = {
    "synthetic-1-verified-apertus-inner-v3",
    "synthetic-1-unverified-apertus-inner-v3",
}


def test_corrected_synthetic1_is_distinct_and_available_on_both_sites():
    for release in CORRECTED_SYNTHETIC1:
        for directory in (CLARIDEN_CONFIGS, RCP_CONFIGS):
            path = directory / f"{release}.cfg"
            config = _assignments(path)
            assert config["DATASET_NAME"] == release
            assert config["ID_COLUMN"] == "source_key"
            assert config["TOKEN_MAP_SOURCE_ROOT"] == f"{CLARIDEN_PROCESSED_ROOT}/{release}"
            assert release in config["PATH_TO_RAW_DATASET"]
            assert release in config["PATH_TO_OUTPUT_FOLDER"]
            assert "-v2" not in config["PATH_TO_RAW_DATASET"]
            assert "-v2" not in config["PATH_TO_OUTPUT_FOLDER"]
            assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0
            if directory == RCP_CONFIGS:
                assert config["PATH_TO_RAW_DATASET"] == f"/mloscratch/{release}"
                assert config["TOKENIZATION_LAUNCH_BACKEND"] == "rcp"
