# apertus-tokenizer-development

Preliminary Apertus v2 tokenizers and their Hugging Face config files.

Intrinsic comparison of the four candidates: [REPORT_focus_candidates.md](REPORT_focus_candidates.md).

## Contents

| folder | vocab | post-processor | character |
|---|---|---|---|
| `preliminary_enh/` | 131072 | yes (`<s> … </s>`) | **English-preserving 131k** — English data boosted (~21 GB) + a moderate EU data boost + an Arabic data/ratio fix. English is slightly denser than the eng5g baseline, EU is improved, Arabic is near Apertus-v1 parity, Chinese ~v1. (`engfull_eu3`) |
| `preliminary_euh/` | 131072 | yes | **EU-dense 131k** — French/German data share boosted (plus a European data boost) with Sinotibetan cut: EU6/EU9 are *denser than Apertus v1* and French/German much improved, at the cost of Chinese and the long tail. (`frde2`) |
| `preliminary_mul/` | 131017 | no | **Balanced consv2 baseline 131k** — no EU boost, no post-processor. (`consv2`) |
| `preliminary_mul_200k/` | 200000 | yes | **200k all-rounder** — denser than Apertus v1 on EU, Chinese, Hindi and Arabic while holding English near parity; the larger vocab removes the head-vs-tail trade-off forced at 131k. (`eusino_v2c`) |

Build recipes (variant keys in `train_tokenizer.py`):
- `preliminary_enh`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_engfull_eu3_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_euh`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_frde2_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_mul`: `nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_v131k_sp124_eng5g`
- `preliminary_mul_200k`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_gm120k_v200k_sp124_eng5g` + BOS/EOS post-processor

Each folder contains `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. The same files are on the Hub at `cmeister/apertus_v2_tokenizer` under matching subfolders.

## Common to all

- Byte-level BPE, NFC normalizer.
- 124 special tokens at IDs 0-123: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat tokens `<|system_start|>` … `<|assistant_end|>`, `<|inner_prefix|>`/`<|inner_suffix|>`, `<|tools_prefix|>`/`<|tools_suffix|>`, `<|tool_output_start|>`/`<|tool_output_end|>`, `<|image|>`, `<|audio|>`; `<reflection>`/`</reflection>`; `<think>`/`</think>`; and 100 reserve slots `<SPECIAL_24>` … `<SPECIAL_123>`.
- `bos_token = <s>`, `eos_token = </s>`, `pad_token = <pad>`, `unk_token = <unk>`.
- No `chat_template` yet, so `apply_chat_template` is not available.

## Per-folder differences

| | `preliminary_enh` | `preliminary_euh` | `preliminary_mul` | `preliminary_mul_200k` |
|---|---|---|---|---|
| vocabulary size | 131072 | 131072 | 131017 (target 131072; 55-token `ParityBpeTrainer` shortfall — size the embedding at 131072, top 55 rows unused) | 200000 |
| pre-tokenizer | `clean_multi_plus2_repcap8` | `clean_multi_plus2_repcap8` | `clean_multi_plus3_repcap8` | `clean_multi_plus2_repcap8` |
| post-processor | `<s> $A </s>` | `<s> $A </s>` | empty (none) | `<s> $A </s>` |
| data character | English-boosted, moderate EU | Fr/De-boosted, EU-dense (Chinese cut) | balanced consv2 | English-boosted, EU-heavy, tail-recouped |

## Default encode behavior and caveats (the post-processor folders: `enh`, `euh`, `mul_200k`)

`tok("text")` / `tok.encode("text")` default to `add_special_tokens=True`, so a single sequence is wrapped as `<s> text </s>`. Apertus's own tokenizer prepends only `<s>` (no `</s>`), so this differs — two things to watch:

- **LLM training.** Every default encode adds both `<s>` and `</s>`. When packing or concatenating documents, encode with `add_special_tokens=False` and add the boundaries yourself, and confirm the trainer is not also inserting its own BOS/EOS — otherwise you get repeated or doubled `<s>`/`</s>`.
- **Chat template.** When one is written, do not have it emit `<s>` and then also tokenize with `add_special_tokens=True` — that double-prepends BOS (the `apply_chat_template` double-BOS footgun). Either render the template and tokenize it with `add_special_tokens=False`, or write the template assuming the post-processor already supplies `<s>`/`</s>`.

`preliminary_mul` has no post-processor, so its `tok("text")` returns the content ids with no `<s>`/`</s>`.

## Usage — `transformers` (`AutoTokenizer`)

`AutoTokenizer` reads `tokenizer_config.json` and `special_tokens_map.json`, so the role tokens (bos/eos/pad/unk) are populated.

```python
from transformers import AutoTokenizer

# Local subfolder (run from the repo root):
tok = AutoTokenizer.from_pretrained("preliminary_enh")
# other candidates: "preliminary_euh", "preliminary_mul", "preliminary_mul_200k"

print(tok.vocab_size, len(tok))          # 131072 131072
print(tok.bos_token, tok.bos_token_id)   # <s> 1
print(tok.eos_token, tok.eos_token_id)   # </s> 2

ids = tok("Hello, world!").input_ids
print(ids)                               # [1, 33882, 135, 1825, 124, 2]   (<s> ... </s> added)
print(tok.decode(ids))                              # <s>Hello, world!</s>
print(tok.decode(ids, skip_special_tokens=True))    # Hello, world!
print(tok("Hello, world!", add_special_tokens=False).input_ids)  # [33882, 135, 1825, 124]
```

Loading the same files from the Hub instead of a local checkout:

```python
tok = AutoTokenizer.from_pretrained("cmeister/apertus_v2_tokenizer", subfolder="preliminary_enh")
```

## Usage — `tokenizers` (`Tokenizer`)

The raw `Tokenizer` reads only `tokenizer.json`. It gives the encoder/decoder, the special tokens, and the post-processor, but not the role mappings or any chat template — use `AutoTokenizer` for those.

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("preliminary_enh/tokenizer.json")

enc = tok.encode("Hello, world!")          # add_special_tokens defaults to True
print(enc.ids)                  # [1, 33882, 135, 1825, 124, 2]
print(enc.tokens)               # ['<s>', 'Hello', ',', 'Ġworld', '!', '</s>']
print(tok.decode(enc.ids))      # Hello, world!   (raw decode skips specials by default)

enc = tok.encode("Hello, world!", add_special_tokens=False)
print(enc.tokens)               # ['Hello', ',', 'Ġworld', '!']

# special tokens are atomic single ids:
enc = tok.encode("<|user_start|>hi<|user_end|>", add_special_tokens=False)
print(enc.tokens)               # ['<|user_start|>', 'hi', '<|user_end|>']
```

## Status

Preliminary. `preliminary_enh`, `preliminary_euh`, and `preliminary_mul_200k` prepend `<s>` and append `</s>` via their post-processor (when `add_special_tokens=True`); `preliminary_mul` does not. The `chat_template` is still not written, so `apply_chat_template` is unavailable.
