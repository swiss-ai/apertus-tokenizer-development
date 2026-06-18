# apertus-tokenizer-development

Preliminary Apertus v2 tokenizers and their Hugging Face config files.

## Contents

| path | build recipe |
|------|--------------|
| `preliminary_euh/` | **EU-data heavy** — trained on a corpus weighted toward European data (European language families re-sourced from FineWeb-2 `quality_10` to ~6 GB each, on top of consv2 ratios + eng5g), so it is denser on EU languages and sparser on the long tail (e.g. Chinese) than a balanced multilingual tokenizer. 90k global merges, `clean_multi_plus2_repcap8` pre-tokenizer, `<s> … </s>` post-processor (`nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eudata_gm90k_v131k_sp124_eng5g_bospost`) |
| `preliminary_mul/` | consv2 ratios + eng5g, no tailcuts, `clean_multi_plus3_repcap8` (`nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_v131k_sp124_eng5g`) |

Each folder contains `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. The same files are on the Hub at `cmeister/apertus_v2_tokenizer` under matching subfolders.

## Common to both

- Byte-level BPE, NFC normalizer.
- 124 special tokens at IDs 0-123: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat tokens `<|system_start|>` … `<|assistant_end|>`, `<|inner_prefix|>`/`<|inner_suffix|>`, `<|tools_prefix|>`/`<|tools_suffix|>`, `<|tool_output_start|>`/`<|tool_output_end|>`, `<|image|>`, `<|audio|>`; `<reflection>`/`</reflection>`; `<think>`/`</think>`; and 100 reserve slots `<SPECIAL_24>` … `<SPECIAL_123>`.
- `bos_token = <s>`, `eos_token = </s>`, `pad_token = <pad>`, `unk_token = <unk>`.
- No `chat_template` yet, so `apply_chat_template` is not available.

## Per-folder differences

| | `preliminary_euh` | `preliminary_mul` |
|---|---|---|
| data mix | **EU-data heavy** — European families re-sourced to ~6 GB each (denser EU, sparser long tail) | balanced consv2 — European families at their natural ~2–3 GB, no EU boost |
| vocabulary size | 131072 (exact) | 131017 (target 131072; 55-token `ParityBpeTrainer` shortfall — size the embedding at 131072, the top 55 rows stay unused) |
| pre-tokenizer | `clean_multi_plus2_repcap8` | `clean_multi_plus3_repcap8` |
| post-processor | `<s> $A </s>` — `<s>`/`</s>` are added automatically when `add_special_tokens=True` (the default) | empty — no `<s>`/`</s>` added on `encode` |

## Default encode behavior and caveats (`preliminary_euh`)

`tok("text")` / `tok.encode("text")` default to `add_special_tokens=True`, so a single sequence is wrapped as `<s> text </s>`. Apertus's own tokenizer prepends only `<s>` (no `</s>`), so this differs — two things to watch:

- **LLM training.** Every default encode adds both `<s>` and `</s>`. When packing or concatenating documents, encode with `add_special_tokens=False` and add the boundaries yourself, and confirm the trainer is not also inserting its own BOS/EOS — otherwise you get repeated or doubled `<s>`/`</s>`.
- **Chat template.** When one is written, do not have it emit `<s>` and then also tokenize with `add_special_tokens=True` — that double-prepends BOS (the `apply_chat_template` double-BOS footgun). Either render the template and tokenize it with `add_special_tokens=False`, or write the template assuming the post-processor already supplies `<s>`/`</s>`.

## Usage — `transformers` (`AutoTokenizer`)

`AutoTokenizer` reads `tokenizer_config.json` and `special_tokens_map.json`, so the role tokens (bos/eos/pad/unk) are populated.

```python
from transformers import AutoTokenizer

# Local subfolder (run from the repo root):
tok = AutoTokenizer.from_pretrained("preliminary_euh")
# the other candidate:
# tok = AutoTokenizer.from_pretrained("preliminary_mul")

print(tok.vocab_size, len(tok))          # 131072 131072
print(tok.bos_token, tok.bos_token_id)   # <s> 1
print(tok.eos_token, tok.eos_token_id)   # </s> 2

ids = tok("Hello, world!").input_ids
print(ids)                               # [1, 49816, 135, 2818, 124, 2]   (<s> ... </s> added)
print(tok.decode(ids))                              # <s>Hello, world!</s>
print(tok.decode(ids, skip_special_tokens=True))    # Hello, world!
print(tok("Hello, world!", add_special_tokens=False).input_ids)  # [49816, 135, 2818, 124]
```

Loading the same files from the Hub instead of a local checkout:

```python
tok = AutoTokenizer.from_pretrained("cmeister/apertus_v2_tokenizer", subfolder="preliminary_euh")
```

`preliminary_mul` has an empty post-processor, so its `tok("Hello, world!").input_ids` returns the content ids with no `<s>`/`</s>`.

## Usage — `tokenizers` (`Tokenizer`)

The raw `Tokenizer` reads only `tokenizer.json`. It gives the encoder/decoder, the special tokens, and the post-processor, but not the role mappings or any chat template — use `AutoTokenizer` for those.

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("preliminary_euh/tokenizer.json")

enc = tok.encode("Hello, world!")          # add_special_tokens defaults to True
print(enc.ids)                  # [1, 49816, 135, 2818, 124, 2]
print(enc.tokens)               # ['<s>', 'Hello', ',', 'Ġworld', '!', '</s>']
print(tok.decode(enc.ids))      # Hello, world!   (raw decode skips specials by default)

enc = tok.encode("Hello, world!", add_special_tokens=False)
print(enc.tokens)               # ['Hello', ',', 'Ġworld', '!']

# special tokens are atomic single ids:
enc = tok.encode("<|user_start|>hi<|user_end|>", add_special_tokens=False)
print(enc.tokens)               # ['<|user_start|>', 'hi', '<|user_end|>']
```

## Status

Preliminary. `preliminary_euh` now prepends `<s>` and appends `</s>` via its post-processor (when `add_special_tokens=True`); `preliminary_mul` does not add them. The `chat_template` is still not written, so `apply_chat_template` is unavailable.
