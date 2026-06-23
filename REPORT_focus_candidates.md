# Focus candidates

Comparison of the four preliminary Apertus v2 tokenizers in this repository
(`preliminary_mul`, `preliminary_enh`, `preliminary_euh`, `preliminary_mul_200k`),
against the current production tokenizer **Apertus v1** and OpenAI's **o200k**
(GPT-4o) as an external 200k reference. See [README.md](README.md) for the build
recipes and usage; intrinsic metrics here were computed with the SwissAI TokEval
library.

The four candidates and their data character:

| Candidate | vocab | character | source variant |
|---|---|---|---|
| `preliminary_mul_200k` **(recommended)** | **200000** | highest European and highest overall compression; compresses both high-resource and low-resource languages more than the 131k tokenizers | `eusino_v2c_frde_kr120` |
| `preliminary_mul` | 131072 | most balanced and fairest; highest compression on Indic languages, Chinese, and the low-resource tail | `consv2_reparam` |
| `preliminary_enh` | 131072 | highest English compression of the 131k tokenizers | `engfull_eu3` |
| `preliminary_euh` | 131072 | highest European compression of the 131k tokenizers; compresses Chinese less | `frde2` |

## Primary recommendation

I recommend **`preliminary_mul_200k`** as the primary candidate. It compresses
both the high-resource European languages and the low-resource tail more than the
131k candidates, which each gain on one and lose on the other. It has the highest
European compression of the set (EU average 4.245 bytes/token on FLORES, against
3.865 for Apertus v1, +9.8%), the highest FLORES sentences-per-token (0.0239),
and the smallest worst-language penalty: the worst-served language needs 3.61x as
many tokens as English on the same text, against 14.70x for Apertus v1. English
compression is close to the other candidates (FineWeb-Edu 4.510 bytes/token,
1.8% below Apertus v1). German compresses more than under Apertus v1 and French
is about the same.

The cost is the vocabulary size. At 200000 it is 53% larger than the 131072 of
Apertus v1 and the other three candidates, which enlarges the embedding and
output tables by the same proportion, with the parameter count and memory that
implies. A larger vocabulary raises compression across the board, so the numbers
above are not a like-for-like comparison with the 131k candidates; the
size-matched comparison is OpenAI's o200k (also 200000). Against o200k it
compresses English about 6% less but compresses the low-resource tail far more
(Tibetan 0.0178 vs 0.0048 sentences per token), is roughly 2x fairer across 205
languages, and has 17 junk tokens against 255. Its lower vocabulary-utilization
figures are the same size effect (more slots, so a smaller used fraction), not a
defect. It has the same template processing and special tokens as the other three
candidates.

If a 131k vocabulary is required (to match Apertus v1's embedding table), the
three 131k candidates each lead on one axis:

- **`preliminary_euh`** has the highest European compression at 131k (EU average
  3.937 bytes/token, +4.0% vs Apertus v1; German 4.332 vs 4.238 chars/token). It
  compresses Chinese 17% less than Apertus v1 (Mandarin 0.917 vs 1.108) and is
  the least fair of the four (Gini 0.138 / 0.122 on FLORES60 / FLORES200, still
  fairer than Apertus v1). The fit when European compression is the priority and
  Chinese can be deprioritized.
- **`preliminary_mul`** is the fairest of the four (Gini 0.088 / 0.105) and most
  balanced: the highest compression on Indic languages (chars/token 2.776),
  Chinese (1.329), and the low-resource tail of the 131k candidates. The fit when
  broad multilingual fairness, rather than European compression, is the goal.
- **`preliminary_enh`** compresses English the most of the 131k candidates
  (FineWeb-Edu 4.486 bytes/token, 2.4% below Apertus v1) and keeps most of the
  multilingual and fairness gains. European languages compress less than under
  Apertus v1. The fit when English compression is the priority.

No language model has been trained on any of these four tokenizers yet, so the
extrinsic section (§6) is pending for all of them.

Compression cells in §1 show `(% diff vs Apertus v1)`. Higher sent/tok and
higher bytes/token are better.

## 1. Compression: four corpora

`sent/tok` is FLORES sentences (lines) per token (higher = more compressed).
`b/t` is bytes per token (higher = more compressed). FineWeb2-proportional is a
7.5 MB multilingual sample (seed 0) drawn across 22 families with per-family
shares matching the natural FineWeb2 byte distribution; all tokenizers measured
on the identical sample. `FLORES EU b/t` is bytes/token over the ten EU languages
(deu/fra/spa/ita/por/nld/swe/pol/ron/dan) on FLORES.

| Tokenizer | FLORES60 sent/tok ↑ | FLORES200 sent/tok ↑ | FineWeb-Edu English b/t ↑ | FineWeb2-proportional b/t ↑ | FLORES EU b/t ↑ |
|---|---|---|---|---|---|
| Apertus v1 | 0.0198 | 0.0142 | 4.595 | 3.077 | 3.865 |
| preliminary_mul | 0.0235 (+18.7%) | 0.0202 (+42.3%) | 4.333 (−5.7%) | **3.796 (+23.4%)** | 3.780 (−2.2%) |
| preliminary_enh | 0.0223 (+12.6%) | 0.0199 (+40.1%) | 4.486 (−2.4%) | 3.621 (+17.7%) | 3.850 (−0.4%) |
| preliminary_euh | 0.0219 (+10.6%) | 0.0195 (+37.3%) | 4.424 (−3.7%) | 3.559 (+15.7%) | 4.041 (+4.6%) |
| **preliminary_mul_200k** | **0.0239 (+20.7%)** | **0.0207 (+45.8%)** | 4.510 (−1.8%) | 3.791 (+23.2%) | **4.245 (+9.8%)** |
| *o200k (200k ref)* | 0.0239 | 0.0176 | **4.786** | 3.533 | 4.040 (+4.5%) |

All four candidates compress the multilingual sets much more than Apertus v1,
and compress English a few percent less. Among the 131k candidates,
`preliminary_enh` has the smallest English loss (−2.4% vs Apertus). The 200k
candidate has the highest FLORES numbers. o200k (same vocabulary size)
compresses English more (4.786) and the full 205-language set much less (0.0176
vs 0.0208), so the larger vocabulary alone does not give multilingual breadth. On
the EU set, `preliminary_euh` (+4.6%) and `preliminary_mul_200k` (+9.8%)
compress more than Apertus v1; `euh` has the highest EU compression of the 131k
tokenizers.

### 1.1 Language character: FLORES chars/token (content-only; higher = more characters per token, i.e. higher compression rate; % diff vs Apertus v1)

This shows how each tokenizer allocates vocabulary across languages.

| Tokenizer | English | EU-avg | French | German | Italian | Indic | Mandarin | Tibetan |
|---|---|---|---|---|---|---|---|---|
| Apertus v1 | 4.742 | 3.785 | 4.296 | 4.238 | 4.024 | 2.352 | 1.108 | 0.433 |
| preliminary_mul | 4.554 (−4.0%) | 3.676 (−2.9%) | 3.710 (−13.6%) | 3.629 (−14.4%) | 3.846 (−4.4%) | **2.776 (+18.0%)** | **1.329 (+19.9%)** | **2.941 (+579.2%)** |
| preliminary_enh | 4.715 (−0.6%) | 3.748 (−1.0%) | 3.771 (−12.2%) | 3.784 (−10.7%) | 3.899 (−3.1%) | 2.427 (+3.2%) | 1.134 (+2.3%) | 2.489 (+474.8%) |
| preliminary_euh | 4.650 (−1.9%) | 3.937 (+4.0%) | **4.200 (−2.2%)** | **4.332 (+2.2%)** | 4.034 (+0.2%) | 2.328 (−1.0%) | 0.917 (−17.2%) | 2.222 (+413.2%) |
| preliminary_mul_200k | 4.739 (−0.1%) | **4.129 (+9.1%)** | 4.295 (−0.0%) | 4.363 (+3.0%) | **4.239 (+5.4%)** | 2.759 (+17.3%) | 1.149 (+3.7%) | 2.518 (+481.6%) |

EU-avg = deu/fra/spa/ita/por/nld/swe/pol/ron/dan. French, German, and Italian
(Switzerland's national languages) are shown separately. `preliminary_euh` is
the only 131k candidate that compresses more than Apertus v1 on the EU average
(3.937 vs 3.785), German (4.332 vs 4.238), and Italian (4.034 vs 4.024); French
(4.200) is just below Apertus (4.296). Every candidate compresses the
low-resource tail much more than Apertus v1 (Tibetan 2.2 to 2.9 vs 0.433).

Per-language compression across the tokenizers is plotted in
[per_language_compression.svg](per_language_compression.svg) (PNG:
[per_language_compression.png](per_language_compression.png)): one panel per
tokenizer, with bars for 15 languages from the European head to the low-resource
tail, spanning Latin, Cyrillic, Arabic, CJK, Indic, Thai, Tibetan, and the
Turkish/Swahili Latin pair. It uses FLORES sentences per token (parallel
sentences, so comparable across scripts; the chars/token table above is not).
Apertus v1 and o200k compress European text well but fragment the low-resource
tail: Tibetan is 0.003 and 0.005 sentences per token, against about 0.020 for
the candidates. `preliminary_mul_200k` is the most even across the set.

Both plots are produced by `make_per_language_plots.py` from the FLORES
parallel files (997 sentences per language, `add_special_tokens=False`).

## 2. Fairness: Gini coefficient and worst-language factor

Worst-language factor is the multiplicative token-count increase, on the same
parallel FLORES content, between the worst-served language and English.

| Tokenizer | FLORES60 Gini ↓ | FLORES200 Gini ↓ | Worst FLORES200 factor ↓ |
|---|---|---|---|
| Apertus v1 | 0.205 | 0.313 | 14.70× (khm_Khmr) |
| preliminary_mul | **0.088** | **0.105** | 3.63× (sat_Olck) |
| preliminary_enh | 0.121 | 0.114 | 4.46× (sat_Olck) |
| preliminary_euh | 0.138 | 0.122 | 4.67× (sat_Olck) |
| preliminary_mul_200k | 0.118 | 0.115 | **3.61× (sat_Olck)** |
| *o200k (200k ref)* | 0.103 | 0.237 | 13.70× (sat_Olck) |


## 3. Vocabulary utilization and junk tokens

| Tokenizer | FLORES60 vocab util ↑ | FLORES200 vocab util ↑ | Junk tokens (≥8-char decorative runs) ↓ |
|---|---|---|---|
| Apertus v1 | 0.556 | 0.643 | 46 |
| preliminary_mul | **0.639** | **0.847** | 17 |
| preliminary_enh | 0.598 | 0.773 | 17 |
| preliminary_euh | 0.620 | 0.775 | 17 |
| preliminary_mul_200k | 0.545 | 0.729 | 17 |
| *o200k (200k ref)* | 0.475 | 0.590 | 255 |

The 200k candidate's lower utilization (0.545 / 0.729) is the vocabulary-size
effect described above, not waste: it uses more tokens in absolute terms.
o200k has 255 decorative-run/glitch tokens, against 17 for each candidate.

Per-language vocabulary utilization (the raw count of distinct vocabulary ids
used to encode each language's 997-sentence FLORES corpus, not a fraction) is
plotted in
[per_language_vocab_utilization.svg](per_language_vocab_utilization.svg) (PNG:
[per_language_vocab_utilization.png](per_language_vocab_utilization.png)), one
panel per tokenizer over the same 15 languages as the compression plot. On the
low-resource tail the candidates use more distinct ids than Apertus v1, which
means dedicated subword tokens rather than byte fallback: Tibetan 525 to 1300
across the candidates against 230 for Apertus v1, and Tamil 853 to 1697 against
771. `preliminary_mul_200k` uses the most distinct ids on Arabic (6077) and
Hindi (2394) of the candidates, from its larger vocabulary.

## 4. Code-structure metrics

AST full-alignment is the fraction of AST-node spans whose token boundaries
match on both ends across the StarCoder sample; operator isolation is the
fraction of arithmetic operators emitted as standalone tokens.

| Tokenizer | AST full-alignment ↑ | Operator isolation ↑ |
|---|---|---|
| Apertus v1 | 0.488 | 0.373 |
| preliminary_mul | **0.689** | **0.991** |
| preliminary_enh | 0.679 | 0.990 |
| preliminary_euh | 0.682 | 0.990 |
| preliminary_mul_200k | 0.681 | 0.990 |
| *o200k (200k ref)* | 0.463 | 0.354 |

All four candidates align to code structure much better than Apertus v1 and
o200k. Apertus v1 and o200k merge operators into surrounding tokens (operator
isolation 0.35 to 0.37) and align to AST boundaries less often.

## 5. Encode throughput

Single-core encode throughput on the English FineWeb-Edu snippet (the same
1000-document snippet as the compression table in §1), measured through the
Hugging Face `tokenizers` Rust backend (`encode_batch`,
`add_special_tokens=False`, `RAYON_NUM_THREADS=1`), reported as the minimum over
11 timed repeats after warmup. Throughput is input bytes divided by encode time,
with MB = 10^6 bytes of input UTF-8 text. Numbers are single-core on a shared
login node, so absolute values carry roughly 10% run-to-run variance; the
relative ordering is stable. All five tokenizers use `ignore_merges=True`.
Produced by `fineweb_edu_throughput.py`.

| Tokenizer | Vocab | Encode MB/s |
|---|---|---|
| Apertus v1 | 131072 | 3.42 |
| preliminary_mul | 131072 | 3.04 |
| preliminary_enh | 131072 | 3.08 |
| preliminary_euh | 131072 | 2.97 |
| preliminary_mul_200k | 200000 | 3.07 |

The four candidates were changed to `ignore_merges=True` (they previously shipped
with it off, like the rest of the training-library default; Apertus v1 already
had it on). On the candidates this raised single-core encode throughput by 1.11x
to 1.13x and produced identical token ids on both the English snippet (1000
documents) and a 10-language FLORES sample (9970 lines), so it changed encode
speed only, not tokenization. The candidates encode at 2.97 to 3.08 MB/s; the
remaining difference from Apertus v1 (3.42) is the pretokenizer regex, which does
more work per byte than the Apertus v1 pretokenizer. The 200k vocabulary does not
lower throughput here: `preliminary_mul_200k` matches the 131k candidates.

## 6. Extrinsic: 1B-parameter LM

Pending. No language model has been trained on any of the four candidates yet,
so downstream BPB, BLiMP, MGSM, MC-math, GSM8K, HumanEval, and MBPP are not
available. This section will be filled once `-mathcode-scratch` and balanced LM
runs exist for the chosen candidate.

## Takeaways

- **`preliminary_mul_200k`** (recommended): with 53% more vocabulary it
  compresses English about as much as the others (FineWeb-Edu 4.510 bytes/token)
  and has the highest EU compression (EU-avg 4.129, German 4.363, French 4.295
  chars/token) and the smallest worst-language penalty (3.61x), while keeping the
  low-resource tail (Indic 2.759, Mandarin 1.149, Tibetan 2.518 chars/token). It
  compresses both the high-resource and low-resource languages more than the 131k
  candidates do. The cost is the larger embedding/output table (200000 vs 131072).
  Against the size-matched o200k it compresses English about 6% less, is roughly
  2x fairer across 205 languages, compresses the low-resource tail far more, and
  has 17 junk tokens against 255.
- **`preliminary_mul`** (131k): the balanced multilingual choice and the fairest
  candidate (Gini 0.088 / 0.105). Highest compression on Indic (2.776), Mandarin
  (1.329), and Tibetan (2.941) of the 131k set, and highest vocabulary
  utilization (0.847 on FLORES200). It compresses English the least (4.333
  bytes/token, 5.7% below Apertus v1).
- **`preliminary_enh`** (131k): English focus. Highest English compression of the
  131k candidates (4.486 bytes/token, 2.4% below Apertus v1), keeping most of the
  multilingual and fairness gains. Indic and Chinese are lower than
  `preliminary_mul`. EU languages compress less than under Apertus v1.
- **`preliminary_euh`** (131k): highest EU compression of the 131k candidates
  (EU-avg 3.937; German and French much improved). Chinese gets less vocabulary,
  so Mandarin drops to 0.917, **below Apertus v1's 1.108 (−17%)**. By Gini it is
  the least fair of the four candidates. Appropriate if European compression is
  the priority and Chinese can be deprioritized.

Notes: FineWeb2-proportional is a 7.5 MB sample (seed 0; all tokenizers on the
identical sample), so its Apertus value (3.077) differs slightly from earlier
6.3 MB runs (3.061). o200k has its own pretokenizer and no NFC normalizer
(inherent to the comparison). Extrinsic results are pending for all four
candidates.
