# Verify the tokenizer job

Set up the [paired checkouts and prerequisites](README.md#set-up-compatible-code),
then run these checks from the repository root:

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
output. They use local fixtures and execute native workers on the test machine.
The tracked-proposal forwarding check mocks the external ADT preflight/submission
boundary and runs the DPP guard, executor and payload checks locally.

Hosted style and config-file checks use public prerequisites. Hosted paired
integration is disabled unless an owner sets
`DPP_INTEGRATION_ENABLED=true` and supplies the read-only `DPP_READ_TOKEN` secret.
It stays skipped for fork pull requests; an enabled job fails if its credential
is absent. The job checks out the full commit in `dpp-revision.txt`; the default
public repository token cannot read private DPP. A skipped job provides no paired
integration result.
