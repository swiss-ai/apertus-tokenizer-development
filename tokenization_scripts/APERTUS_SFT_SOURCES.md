# Source-separated Apertus SFT tokenization

The original `Apertus-1.5-SFT-mix-pretrain-v1.cfg` remains the immutable recipe
for the already tokenized mixed artifact. For future runs that need independent
weights for each constituent source, first run
`data-pipeline-pretrain/pipelines/apertus-sft-pretrain/partition_sources.py` and
publish its sealed, source-pure Parquet root. Then generate ordinary tokenizer
configs from that manifest:

```bash
python3 tokenization_scripts/generate_apertus_sft_source_configs.py \
  --partition-manifest /capstor/.../Apertus-1.5-SFT-mix-pretrain-v1-by-source/_SOURCE_PARTITION_SUCCESS.json \
  --text-root /capstor/.../Apertus-1.5-SFT-mix-pretrain-v1-by-source \
  --token-root /capstor/.../datasets_tokenized/Apertus-1.5-SFT-mix-pretrain-v1-by-source_apertus_v2 \
  --config-dir /path/to/generated-apertus-sft-configs
```

Every generated config points `PATH_TO_RAW_DATASET` at exactly one
`data/<stable-source-slug>` directory and gives that source a distinct
`DATASET_NAME` and preprocessing-metadata folder. The existing launcher writes
token pairs below `<token-root>/preliminary_mul_200k/<slug>/dump-*`, so each
source has its own sampler-addressable token path. They use the existing `tokenize_script.sh`;
there is no new tokenizer. `_CONFIGS_SUCCESS.json` pins the input partition
seal, template, source labels, config hashes, and output paths. Reruns accept
identical files and reject divergent ones. Review the generated configs before
launching Slurm jobs; their template's reservation and resource settings are
not inferred from the source partition.

For the already sealed mixed token corpus, do **not** run these configs merely
to make new weights possible. A lossless token-pair backfill can copy the
original, **pre-reserve** `.bin` sequences into source-pure `.bin/.idx/.map`
groups, conserving token bytes. Run the ordinary long-context reserve policy
again on the **whole** source-separated token root. The reserve splitter keeps
relative source paths in both kept and held-back outputs, so source roots stay
sampler-addressable without 35 independent reserve jobs. This retains the
existing corpus-level 10% policy; record realized held-back counts per source
before choosing weights. The historical mixed reserve is not reused. The backfill uses
the `PairSplitPlan` code in the separate `lc-reserve` PR and is kept as an
operational script outside this repo because this draft PR is based on a
branch without that splitter. Do not point the sampler at unsealed output.
