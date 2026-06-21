# apertus-tokenizer-development

Preliminary Apertus v2 tokenizers and their Hugging Face config files.

Intrinsic comparison of the four candidates: [REPORT_focus_candidates.md](REPORT_focus_candidates.md).

> **Warning: template processing differs from Apertus v1.** With `add_special_tokens=True`, the post-processor wraps a single sequence as `<s> text </s>`, adding both BOS and EOS. Apertus v1 prepends only `<s>` and adds no `</s>`. This is a deliberate change requested by the engineering team. Configure training, packing, and any chat template accordingly. See the "Default encode behavior and caveats" section below.

## Contents

| folder | vocab | post-processor | character |
|---|---|---|---|
| `preliminary_enh/` | 131072 | yes (`<s> … </s>`) | **English-preserving 131k**: English data boosted (~21 GB) plus a moderate EU data boost and an Arabic data/ratio fix. English is slightly denser than the eng5g baseline, EU is denser, Arabic is near Apertus-v1 parity, Chinese is near v1. (`engfull_eu3`) |
| `preliminary_euh/` | 131072 | yes | **EU-dense 131k**: French/German data share boosted (plus a European data boost) with a Sinotibetan cut. EU6/EU9 are denser than Apertus v1 and French/German are much denser, at the cost of Chinese and the long tail. (`frde2`) |
| `preliminary_mul/` | 131072 | yes | **Balanced multilingual 131k**: consv2 with the reparam ratio adjustment. Denser EU than the plain consv2 base, with the same tail compression and fairness. (`consv2_reparam`) |
| `preliminary_mul_200k/` | 200000 | yes | **200k all-rounder**: denser than Apertus v1 on EU, Chinese, Hindi, and Arabic, with English near parity. The larger vocabulary compresses both high-resource and low-resource languages well, which the 131k tokenizers do not. (`eusino_v2c`) |

Build recipes (variant keys in `train_tokenizer.py`):
- `preliminary_enh`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_engfull_eu3_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_euh`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_frde2_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_mul`: `nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_reparam_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_mul_200k`: `nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_gm120k_v200k_sp124_eng5g` + BOS/EOS post-processor

Each folder contains `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. The same files are on the Hub at `cmeister/apertus_v2_tokenizer` under matching subfolders.

## Common to all

- Byte-level BPE, NFC normalizer.
- 124 special tokens at IDs 0-123: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat tokens `<|system_start|>` … `<|assistant_end|>`, `<|inner_prefix|>`/`<|inner_suffix|>`, `<|tools_prefix|>`/`<|tools_suffix|>`, `<|tool_output_start|>`/`<|tool_output_end|>`, `<|image|>`, `<|audio|>`; `<reflection>`/`</reflection>`; `<think>`/`</think>`; the PII tokens `<pii-iban>`(24), `<pii-email>`(25), `<pii-ip>`(26); and 97 reserve slots `<SPECIAL_27>` … `<SPECIAL_123>`.
- `bos_token = <s>`, `eos_token = </s>`, `pad_token = <pad>`, `unk_token = <unk>`.
- No `chat_template` yet, so `apply_chat_template` is not available.

## Per-folder differences

| | `preliminary_enh` | `preliminary_euh` | `preliminary_mul` | `preliminary_mul_200k` |
|---|---|---|---|---|
| vocabulary size | 131072 | 131072 | 131072 | 200000 |
| pre-tokenizer | `clean_multi_plus2_repcap8` | `clean_multi_plus2_repcap8` | `clean_multi_plus3_repcap8` | `clean_multi_plus2_repcap8` |
| post-processor | `<s> $A </s>` | `<s> $A </s>` | `<s> $A </s>` | `<s> $A </s>` |
| data character | English-boosted, moderate EU | Fr/De-boosted, EU-dense (Chinese cut) | balanced consv2 (reparam) | English-boosted, EU-heavy, tail recovered |

## Default encode behavior and caveats (all four folders)

`tok("text")` / `tok.encode("text")` default to `add_special_tokens=True`, so a single sequence is wrapped as `<s> text </s>`. Apertus's own tokenizer prepends only `<s>` (no `</s>`), so this differs. Two things to watch:

- **LLM training.** Every default encode adds both `<s>` and `</s>`. When packing or concatenating documents, encode with `add_special_tokens=False` and add the boundaries yourself, and confirm the trainer is not also inserting its own BOS/EOS, otherwise you get repeated or doubled `<s>`/`</s>`.
- **Chat template.** When one is written, do not have it emit `<s>` and then also tokenize with `add_special_tokens=True`, which double-prepends BOS (a common `apply_chat_template` mistake). Either render the template and tokenize it with `add_special_tokens=False`, or write the template assuming the post-processor already supplies `<s>`/`</s>`.

All four folders behave identically here: the post-processor adds `<s>`/`</s>` when `add_special_tokens=True`.

## Usage: `transformers` (`AutoTokenizer`)

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

## Usage: `tokenizers` (`Tokenizer`)

The raw `Tokenizer` reads only `tokenizer.json`. It gives the encoder/decoder, the special tokens, and the post-processor, but not the role mappings or any chat template. Use `AutoTokenizer` for those.

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

Preliminary. All four folders prepend `<s>` and append `</s>` via their post-processor (when `add_special_tokens=True`). The `chat_template` is still not written, so `apply_chat_template` is unavailable.
