# Root-aligned tokenization

`preprocess_megatron.py` is the supported job. It reads completed exact DPP root
and primary-text Data, preserves every row in root order, and writes new aligned
token Data plus its automatic Int64 `stored_tokens` Annotation. It neither selects
nor rehydrates rows. Dataset preparation and physical population changes belong
to the preparation producer; the sampler owns selection and its run maps.

## Set up compatible code

Use Python 3.10 or newer and a local DPP checkout at the exact commit in
[`dpp-revision.txt`](dpp-revision.txt). DPP is private: access must be granted by its
owner. Do not copy its source into this public repository or embed credentials.
The paired commit is a local draft until it is published by the owner; neither
repository's default branch is an implicit compatibility guarantee.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r tests/requirements/runtime.txt
# DPP_CHECKOUT is your separately authorized local DPP source checkout.
git -C "$DPP_CHECKOUT" checkout "$(cat tokenization_scripts/dpp-revision.txt)"
export PYTHONPATH="$DPP_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tokenization_scripts.preprocess_megatron --help
```

These public prerequisites support source tests and this local job; they do not
claim a production DPP installation profile. Use the producer's supported runtime
for deployment. Run one coordinator inside an allocated machine; scheduler
submission remains the owning environment's responsibility.

## Run a job

The input must already have complete root-aligned metadata and payloads beneath
one physical `data_root`. Use the exact approved root, text and new token Data
references from that producer's port. Do not turn old `.cfg` names or dump paths
into invented declaration references.

The tokenizer JSON must be a local tracked file in a Git checkout with an origin
URL. Supply the full commit containing those exact bytes. The job compares local
bytes with that committed asset before output mutation, then workers load the
verified bytes offline. A mutable directory label is insufficient. The recorded
encoding context includes repository, commit and relative asset path, along with
explicit disabled postprocessing/truncation/padding. No hashes or new manifest
scheme are introduced.

From this repository root, with shell variables set to your reviewed identities:

```sh
python -m tokenization_scripts.preprocess_megatron \
  --data-root "$DATA_ROOT" \
  --root-ref "$ROOT_REF" --input-ref "$TEXT_REF" --output-ref "$TOKEN_REF" \
  --tokenizer-file "$TOKENIZER_JSON" --tokenizer-revision "$ASSET_COMMIT" \
  --prefix-tokens 1 --suffix-tokens 2 \
  --logging-dir "$TOKEN_LOGS" \
  --tasks 8 --workers 2 --threads 4
```

The shown boundary IDs are examples: check the actual asset and encoding recipe.
Both boundary arguments are required; pass an empty string for no prefix/suffix.
The body uses `add_special_tokens=False`. DPP inserts configured boundaries once.
The primary text field comes from the input Data's `primary_content` semantics;
user IDs are unnecessary for root alignment. Empty text, empty encoded bodies,
wrong output refs, missing input, cardinality/alignment problems and invalid IDs
fail, without dropping a row or hiding errors behind boundary tokens.

`--tasks` counts deterministic native partitions; `--workers` counts concurrent
processes, and `--threads` sets each process's Rayon pool. Defaults are one each.
Budget workers × threads within allocated CPUs. `--batch-size` defaults to 1000
rows and `--batch-bytes` to 33554432 UTF-8 bytes. A single oversize row runs alone
without truncation. `--max-sequences-per-shard` defaults to 100000 index entries.
The output writer uses uint16/code8 for vocabularies through 65536 IDs, otherwise
nonnegative int32/code4. Output sharding may differ from input Parquet sharding.

`run_producer` guards token Data and automatic lengths. Each native task writes
BIN/IDX and lengths, including explicit empty pairs for idle tasks. Finalization
checks native completion, exact pair coverage, token IDs/boundaries, lengths and
corruption before publishing both records. Token sequences correspond directly
to root ordinals. No tokenizer `.map` is created or read; historical artifacts
and maps remain with their original producer commits.

## Retry and verify

Nonempty outputs fail by default. Add `--resume` only when inputs, verified asset,
boundaries, partitioning, resources and native logs are compatible, and no readers
or competing writers are active. Native completed tasks are reused; payloads are
qualified again. The operator remains responsible for resume compatibility.
The held completion/lifecycle policy gap remains open; this job adds no lifecycle
policy and does not claim to seal releases. ADT is disabled and no declaration is
published. Payload readiness is separate from eligibility and trainer consumption.

Run owning checks against the paired actual source interface:

```sh
export MYPYPATH="$DPP_CHECKOUT/src"
python -m mypy --config-file pyproject.toml
ruff check tokenization_scripts/preprocess_megatron.py tests/test_tokenization_scripts.py
ruff format --check tokenization_scripts/preprocess_megatron.py tests/test_tokenization_scripts.py
python -m pytest -q
```

The workflow tests construct a tiny local Git-pinned tokenizer/root, execute native
workers fresh and resumed, verify exact sequences/lengths and uint16/int32 dtypes,
assert no maps, and reject modified assets, missing roots, empty bodies and corrupt
output. They download no model, submit no scheduler jobs and contact no ADT service.
Hosted style checks use public prerequisites. Hosted paired integration is explicitly
skipped unless an owner supplies the read-only `DPP_READ_TOKEN` secret; the default
public repository token cannot read private DPP. A skipped job is not integration
evidence, and local source checks do not establish hosted CI or production install
success.

## Dataset adoption and historical recipes

All 53 retained `.cfg` files (52 in `configs_apertus_v2/` and
`example_config.cfg`) require migration. Their exact filenames and original bytes
remain in this tree; the last supported dump runtime is recoverable at
`28ad57a2757f6f72edb0def57ee725b6812f2df3`. This includes the four released
CodeAlchemy configurations; their source-specific filtered-population checks remain.
No retained config is accepted by the new CLI. The old `tokenize_script.sh`,
`tokenize.sh`, dump preparation/size helpers and container launcher recipe are
retired from the supported workflow.

Each producer must establish actual root/input refs, preserve its real preparation,
source approvals and exclusions, reconcile physical rehydration/filtering and source
lineage, and choose a distinct output identity with the exact encoding recipe.
Then record the reviewed command with that producer's adoption change. Keep actual
frozen producing commits, payloads and maps; a later runtime merge cannot replace
the identity that produced them. This paired draft remains open until compatible
DPP publication/access, producer-specific adoption and the held lifecycle decisions
are resolved.
