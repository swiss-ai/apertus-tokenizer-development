# Verify the tokenizer job

Set up the [paired checkouts and prerequisites](README.md#set-up-compatible-code),
then run these checks from the repository root:

```sh
export MYPYPATH="$DPP_CHECKOUT/src"
python -m mypy --config-file pyproject.toml
ruff check tokenization_scripts tests
ruff format --check tokenization_scripts tests
python -m pytest -q -ra
```

The workflow tests create a Git-pinned tokenizer and a text root through
Datatrove execution. They run fresh jobs, resume interrupted finalization and
check immutable completion, exact token cells, stored lengths and root alignment.
They preserve empty text with boundaries and reject asset drift, lost rows and
corrupt Parquet.
The public `python -m` entry point runs with one and two spawned workers and
explicit boundaries; fixtures exercise vocabularies on both sides of 65536 IDs.

Encoding-step tests cover independent row and UTF-8 byte limits, Unicode and
oversize rows, exact cardinality and preservation of documents and coordinates.
Fast and fallback tokenizers produce the same body IDs. Pickle, dill and deep-copy
transport retain verified bytes while each worker initializes its tokenizer;
executor logs omit the model bytes.

Run the checks on Python 3.10 and 3.12 with the exact paired DPP commit in
`dpp-revision.txt`. The configured pytest command discovers the complete test
suite.
