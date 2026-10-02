# Source-separated Apertus SFT tokenization

`generate_apertus_sft_source_configs.py` writes one ordinary `tokenize_script.sh`
config per source of the Apertus 1.5 SFT pretraining text. Its input is the sealed,
source-pure Parquet root written by data-pipeline-pretrain's
`pipelines/apertus-sft-pretrain/main_3_partition_sources.py`.

## Generate the configs

```bash
python3 tokenization_scripts/generate_apertus_sft_source_configs.py \
  --partition-manifest /capstor/.../Apertus-1.5-SFT-mix-pretrain-v1-by-source/_SOURCE_PARTITION_SUCCESS.json \
  --text-root /capstor/.../Apertus-1.5-SFT-mix-pretrain-v1-by-source \
  --token-root /capstor/.../datasets_tokenized/Apertus-1.5-SFT-mix-pretrain-v1-by-source_apertus_v2 \
  --config-dir <config-dir>
```

| argument | default | meaning |
|---|---|---|
| `--partition-manifest` | required | `_SOURCE_PARTITION_SUCCESS.json` seal. It must be byte-identical to the copy in `--text-root`. |
| `--text-root` | required | Absolute source-partition root containing `data/<slug>/part-*.parquet`. |
| `--token-root` | required | Absolute token output root. |
| `--template` | `tokenization_scripts/configs_apertus_v2/Apertus-1.5-SFT-mix-pretrain-v1.cfg` | Config copied for every source. It must assign the six keys below, which are replaced; every other line, including the Slurm settings, is copied unchanged. |
| `--config-dir` | required | Receives one `<slug>.cfg` per source and `_CONFIGS_SUCCESS.json`. |

Each generated config sets:

| key | value |
|---|---|
| `DATASET_NAME` | `<slug>` |
| `PATH_TO_RAW_DATASET` | `<text-root>/data/<slug>` |
| `PATH_TO_PREPROCESSING_METADATA` | `<token-root>/_preprocessing/<slug>` |
| `PATH_TO_OUTPUT_FOLDER` | `<token-root>` |
| `DUMPS_NUMBER` | 1 below 100,000 rows, 4 below 500,000 rows, otherwise 8; never more than the source's Parquet parts |
| `NUMBER_OF_DATATROVE_TASKS` | ceil(rows / (dumps * 5,000)), between 1 and 16 |

`_CONFIGS_SUCCESS.json` records the SHA-256 of the partition seal, of the template and
of every generated config, plus each source's label, rows, parts and job shape. A rerun
accepts files identical to the existing ones and fails on any difference.

## Tokenize

Review the generated configs, then submit each one from `tokenization_scripts/`, which
the launcher requires as its working directory:

```bash
cd tokenization_scripts
./tokenize_script.sh <config-dir>/<slug>.cfg
```

Token pairs land in `<token-root>/preliminary_mul_200k/<slug>/dump-*`.
