# Verify the tokenizer job

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
The tracked-proposal forwarding check replaces only the external ADT preflight/
submission boundary while running the real DPP guard, executor and payload checks;
it does not claim a live catalogue submission.
Hosted style and historical config checks use public prerequisites. Hosted paired
integration is disabled unless an owner sets
`DPP_INTEGRATION_ENABLED=true` and supplies the read-only `DPP_READ_TOKEN` secret.
It stays skipped for fork pull requests; an enabled job fails if its credential
is absent. The job checks out the full commit in `dpp-revision.txt`; the default
public repository token cannot read private DPP. A skipped job is not integration
evidence, and local source checks do not establish hosted CI or production install
success.
