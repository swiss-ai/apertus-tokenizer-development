# Corrected code and SYNTHETIC-1 inputs

The paired Clariden/RCP configs select seven corrected code/agent releases
(`*-v4`, OpenHands `*-v5`) and both SYNTHETIC-1 `*-v5` populations. Each consumes
the complete prepared population, uses `text` and `source_key`, and records the
durable corrected prepared root in token maps. Historical configs remain available.

Use the [ordinary preparation, worker and validation workflow](../README.md).
Pass the source-processing report to `validate_stem_direct.py --processing-report`;
configuration files alone do not establish payload acceptance.

CPU settings are explicit: zero GPUs, one worker, 16 tokenizer threads, a 128-document
batch ceiling and a 32 MiB input-byte ceiling. RCP configs select the existing RCP
backend; only prepared-read/output paths differ from their Clariden counterparts.
Qualify dump count and concurrency against the actual prepared file inventory.
