# apertus-tokenizer-development

Preliminary Apertus v2 tokenizers and their Hugging Face config files.

## Contents

| path | build recipe |
|------|--------------|
| `preliminary_enh/` | consv2 ratios + tailcuts overlay + eng5g, 90k global merges (`nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_tailcuts_gm90k_v131k_sp124_eng5g_fr`) |
| `preliminary_mul/` | consv2 ratios + eng5g, no tailcuts (`nfc_clean_multi_plus3_repcap8_capped_hybrid_window_tuned_consv2_v131k_sp124_eng5g`) |

Each folder contains `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`. The same files are on the Hub at `cmeister/apertus_v2_tokenizer` under matching subfolders.

Shared properties:

- Byte-level BPE, NFC normalizer, `clean_multi_plus3_repcap8` pre-tokenizer.
- Vocabulary size 131017. The target was 131072; the 55-token shortfall is a `ParityBpeTrainer` accounting effect. Size the model embedding at 131072 and the top 55 rows stay unused.
- 124 special tokens at IDs 0-123: `<unk>`(0), `<s>`(1), `</s>`(2), `<pad>`(3); the chat tokens `<|system_start|>` … `<|assistant_end|>`, `<|inner_prefix|>`/`<|inner_suffix|>`, `<|tools_prefix|>`/`<|tools_suffix|>`, `<|tool_output_start|>`/`<|tool_output_end|>`, `<|image|>`, `<|audio|>`; `<reflection>`/`</reflection>`; `<think>`/`</think>`; and 100 reserve slots `<SPECIAL_24>` … `<SPECIAL_123>`.
- `bos_token = <s>`, `eos_token = </s>`, `pad_token = <pad>`, `unk_token = <unk>`.
- Empty post-processor: special tokens are not added automatically (`add_bos_token = false`). A leading `<s>` is not prepended on `encode`; it has to come from a chat template or be added by the caller.
- No `chat_template` yet, so `apply_chat_template` is not available.

## Usage — `transformers` (`AutoTokenizer`)

`AutoTokenizer` reads `tokenizer_config.json` and `special_tokens_map.json`, so the role tokens (bos/eos/pad/unk) are populated.

```python
from transformers import AutoTokenizer

# Local subfolder (run from the repo root):
tok = AutoTokenizer.from_pretrained("preliminary_enh")
# the other candidate:
# tok = AutoTokenizer.from_pretrained("preliminary_mul")

print(tok.vocab_size, len(tok))          # 131017 131017
print(tok.bos_token, tok.bos_token_id)   # <s> 1
print(tok.eos_token, tok.eos_token_id)   # </s> 2

ids = tok("Hello, world!").input_ids
print(ids)                               # [31873, 135, 1966, 124]
print(tok.decode(ids))                   # Hello, world!   (no <s> / </s> added)
```

Loading the same files from the Hub instead of a local checkout:

```python
tok = AutoTokenizer.from_pretrained("cmeister/apertus_v2_tokenizer", subfolder="preliminary_enh")
```

## Usage — `tokenizers` (`Tokenizer`)

The raw `Tokenizer` reads only `tokenizer.json`. It gives the encoder/decoder and the special tokens, but not the role mappings or any chat template — use `AutoTokenizer` for those.

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("preliminary_enh/tokenizer.json")

enc = tok.encode("Hello, world!")
print(enc.ids)                  # [31873, 135, 1966, 124]
print(enc.tokens)               # ['Hello', ',', 'Ġworld', '!']
print(tok.decode(enc.ids))      # Hello, world!

# special tokens are atomic single ids:
enc = tok.encode("<|user_start|>hi<|user_end|>", add_special_tokens=False)
print(enc.tokens)               # ['<|user_start|>', 'hi', '<|user_end|>']
```

## Status

These are preliminary. Two items are not finalized: whether the post-processor should auto-prepend `<s>` (it currently does not), and the chat template (not yet written).
