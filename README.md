# apertus-tokenizer-development

Preliminary Apertus v2 tokenizers and their Hugging Face config files.

Documentation:

- [REPORT_focus_candidates.md](REPORT_focus_candidates.md): the four-way comparison against Apertus v1 and o200k, with the recommendation.
- [DEVELOPMENT_RECORD.md](DEVELOPMENT_RECORD.md): the development record (design ablations, downstream-LM evidence, reference-tokenizer comparison, production safety).
- [TRAINING.md](TRAINING.md): training recipes, data, and reproduction steps.
- [apertus_tokenizer_design.md](apertus_tokenizer_design.md): pretokenization and special-token design rationale.

**Recommended candidate: `preliminary_mul_200k`** (200064 vocabulary, 128-aligned). It is European-focused with substantial broad multilingual coverage: it has the highest European compression of the tokenizers here, and it also compresses the low-resource languages more than the 131k tokenizers, which each improve one and worsen the other. It has the smallest worst-language penalty of the set. The larger vocabulary means a 53% larger embedding and output table than the 131072-vocabulary options (Apertus v1 and the other three candidates). Use one of the 131k candidates instead if you need to match Apertus v1's vocabulary size.

> **Warning: template processing differs from Apertus v1.** With `add_special_tokens=True`, the post-processor wraps a single sequence as `<s> text </s>`, adding both BOS and EOS. Apertus v1 prepends only `<s>` and adds no `</s>`. This is a deliberate change requested by the engineering team. Configure training, packing, and any chat template accordingly. See the "Default encode behavior and caveats" section below.

## Contents

| tokenizer | vocab | post-processor | character |
|---|---|---|---|
| `preliminary_mul_200k/` | 200064 | yes (`<s> … </s>`) | **Recommended.** European-focused with broad multilingual coverage: the highest European compression of the set and the smallest worst-language penalty, and it compresses the low-resource languages more than the 131k tokenizers. |
| `preliminary_mul/` | 131072 | yes | Most balanced and fairest of the four. Highest compression on Indic languages, Chinese, and the low-resource tail among the 131k tokenizers; compresses English the least. |
| `preliminary_enh/` | 131072 | yes | Highest English compression at 131k, trained with more English data, while keeping most of the multilingual and fairness gains. |
| `preliminary_euh/` | 131072 | yes | Highest European compression at 131k. Trained with more French and German data and less Chinese data; compresses Chinese less than Apertus v1 and is the least fair of the four. |

The numbers behind these characterizations are in [REPORT_focus_candidates.md](REPORT_focus_candidates.md).

Build recipes (variant keys in `train_tokenizer.py`):
- `preliminary_enh`: `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_engfull_eu3_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_euh`: `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_frde2_gm110k_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_mul`: `pa_bpe_nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_reparam_v131k_sp124_eng5g` + BOS/EOS post-processor
- `preliminary_mul_200k`: `pa_bpe_nfc_clean_multi_plus2_repcap8_capped_hybrid_window_tuned_consv2_eusino_v2c_frde_kr120_gm130k_v200064_sp124_eng5g` + BOS/EOS post-processor (vocab 200064, 128-aligned; IDs 0-199999 identical to the prior 200000 build)

Full training recipes, environment, data, and reproduction steps: [TRAINING.md](TRAINING.md).

Each folder contains `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.

## Common to all

- Byte-level BPE, NFC normalizer.
- 124 special tokens at IDs 0-123: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat tokens `<|system_start|>` … `<|assistant_end|>`, `<|inner_prefix|>`/`<|inner_suffix|>`, `<|tools_prefix|>`/`<|tools_suffix|>`, `<|tool_output_start|>`/`<|tool_output_end|>`, `<|image|>`, `<|audio|>`; `<reflection>`/`</reflection>`; `<think>`/`</think>`; the PII tokens `<iban-pii>`(24), `<email-pii>`(25), `<ip-pii>`(26); and 97 reserve slots `<SPECIAL_27>` … `<SPECIAL_123>`.
- `bos_token = <s>`, `eos_token = </s>`, `pad_token = <pad>`, `unk_token = <unk>`.
- No `chat_template` yet, so `apply_chat_template` is not available.

## Per-tokenizer differences

| | `preliminary_enh` | `preliminary_euh` | `preliminary_mul` | `preliminary_mul_200k` |
|---|---|---|---|---|
| vocabulary size | 131072 | 131072 | 131072 | 200064 |
| pre-tokenizer | `clean_multi_plus2_repcap8` | `clean_multi_plus2_repcap8` | `clean_multi_plus3_repcap8` | `clean_multi_plus2_repcap8` |
| post-processor | `<s> $A </s>` | `<s> $A </s>` | `<s> $A </s>` | `<s> $A </s>` |
| data character | more English data | more French/German data, less Chinese | balanced multilingual | European-focused with broad multilingual coverage, larger vocabulary |

## Default encode behavior and caveats (all four tokenizers)

`tok("text")` / `tok.encode("text")` default to `add_special_tokens=True`, so a single sequence is wrapped as `<s> text </s>`. Apertus's own tokenizer prepends only `<s>` (no `</s>`), so this differs. Two things to watch:

- **LLM training.** Every default encode adds both `<s>` and `</s>`. When packing or concatenating documents, encode with `add_special_tokens=False` and add the boundaries yourself, and confirm the trainer is not also inserting its own BOS/EOS, otherwise you get repeated or doubled `<s>`/`</s>`.
- **Chat template.** When one is written, do not have it emit `<s>` and then also tokenize with `add_special_tokens=True`, which double-prepends BOS (a common `apply_chat_template` mistake). Either render the template and tokenize it with `add_special_tokens=False`, or write the template assuming the post-processor already supplies `<s>`/`</s>`.

All four tokenizers behave identically here: the post-processor adds `<s>`/`</s>` when `add_special_tokens=True`.

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

Preliminary. All four tokenizers prepend `<s>` and append `</s>` via their post-processor (when `add_special_tokens=True`). The `chat_template` is still not written, so `apply_chat_template` is unavailable.
