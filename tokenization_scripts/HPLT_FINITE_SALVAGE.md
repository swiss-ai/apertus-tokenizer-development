# HPLT finite-score recovery configs

The four `configs_apertus_v2/hplt-ia-finite-score-salvage-*.cfg` files tokenize
the eight recovered language populations from [pipeline #64](https://github.com/swiss-ai/data-pipeline-pretrain/pull/64).
They read its sealed examples manifest, preserve document IDs and row maps, and
write a new token release. Other languages and English reuse their existing
token releases. Finite embedding corruption remains eligible under the approved
provisional selection policy.

Use the [standard preparation, worker and validation entry points](README.md).
The configs group by language and use 4-GiB input dumps, up to 32 workers with
four threads each, and batches bounded by 128 documents and 32 MiB.

Required source-map/LC runtime: data-pipeline-pretrain commit
`fc1e5a86526e38c5f31aa6a099473f5a95b511e7`, with Datatrove 0.6.0 and Arrow 22.0.0.
Required `preliminary_mul_200k/tokenizer.json` SHA-256:
`cd403d3f219e2433e3f78b32644b8e6a6134668e15138e6546360330635a96b9`.

Run workers inside an allocated Clariden EDF container with the runtime's `src`
on `PYTHONPATH` and `TOKENIZATION_LAUNCH_BACKEND=rcp` for direct execution.
Freeze the prepared dump inventory before starting and check it before retries;
assign each pending dump once. Strict validation precedes the existing LC reserve
workflow (fraction 0.10, seed 20260907, length-buckets-v3).

Input-pin checks and the kept-token union are owned by
`pipelines/hplt-4.0/main_6_token_release.py` in data-pipeline-pretrain.
Follow that pipeline's **Token-release handoff** runbook for the commands and
required paths. It validates and links unchanged and recovered kept pairs under
a fresh identity, excluding reserved tokens and preserving available old maps.
For replay, copy the config and choose fresh output/metadata/control roots.
Use measured union counts for sampler adoption.
