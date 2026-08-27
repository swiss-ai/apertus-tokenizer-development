# Focus candidates

Comparison of the four preliminary Apertus v2 tokenizers in this repository
(`preliminary_mul`, `preliminary_enh`, `preliminary_euh`, `preliminary_mul_200k`),
against the current production tokenizer **Apertus v1** and OpenAI's **o200k**
(GPT-4o) as an external 200k reference; o200k has its own pretokenizer and no NFC
normalizer, inherent to the comparison. See [README.md](README.md) for the build
recipes and usage; intrinsic metrics here were computed with the SwissAI TokEval
library.

The four candidates and their data character:

| Candidate | vocab | character | source variant |
|---|---|---|---|
| `preliminary_mul_200k` **(recommended)** | **200064** | European-focused with broad multilingual coverage: highest European compression, and compresses the low-resource languages more than the 131k tokenizers | `eusino_v2c_frde_kr120` |
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
1.8% below Apertus v1). It compresses German more than Apertus v1, and French
about the same.

At 200064, the vocabulary is 53% larger than the 131072 of
Apertus v1 and the other three candidates, which enlarges the embedding and
output tables by the same proportion, with the parameter count and memory that
implies. A larger vocabulary raises compression across the board, so the numbers
above are not a like-for-like comparison with the 131k candidates; the
size-matched comparison is OpenAI's o200k (200000, against this build's 200064). Against o200k it
compresses English about 6% less but compresses the low-resource tail far more
(Tibetan 0.0172 vs 0.0048 sentences per token), is roughly 2x fairer across 205
languages, and has 17 junk tokens against 255. Its lower vocabulary-utilization
figures are the same size effect (more slots, so a smaller used fraction), not a
defect. It has the same template processing and special tokens as the other three
candidates.

If a 131k vocabulary is required (to match Apertus v1's embedding table), the
three 131k candidates each lead on one axis:

- **`preliminary_euh`** has the highest European compression at 131k (EU average
  3.937 chars/token, +4.0% vs Apertus v1; German 4.332 vs 4.238 chars/token). It
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
  multilingual and fairness gains. European languages are compressed less than
  under Apertus v1. The fit when English compression is the priority.

Trained-LM (extrinsic) results for all four candidates are in §6 (with the
tokenizer-identity caveats noted there), and the full downstream panel is in
[DEVELOPMENT_RECORD.md](DEVELOPMENT_RECORD.md).

Compression cells in §1 show `(% diff vs Apertus v1)`. Higher sent/tok and
higher bytes/token are better. In tables, bold marks the best value in a column.

## 1. Compression: four corpora

`sent/tok` is FLORES sentences (lines) per token (higher = more compressed).
`b/t` is bytes per token (higher = more compressed). FineWeb2-proportional is a
6.3 MB multilingual sample (207 files, seed 0) drawn across 22 families with
per-family shares matching the natural FineWeb2 byte distribution; all tokenizers
measured on the identical sample. `FLORES EU b/t` is bytes/token over the ten EU languages
(deu/fra/spa/ita/por/nld/swe/pol/ron/dan) on FLORES.

| Tokenizer | FLORES60 sent/tok ↑ | FLORES200 sent/tok ↑ | FineWeb-Edu English b/t ↑ | FineWeb2-proportional b/t ↑ | FLORES EU b/t ↑ |
|---|---|---|---|---|---|
| Apertus v1 | 0.0198 | 0.0142 | 4.595 | 3.061 | 3.865 |
| preliminary_mul | 0.0235 (+18.7%) | 0.0202 (+42.3%) | 4.333 (−5.7%) | **3.807 (+24.4%)** | 3.780 (−2.2%) |
| preliminary_enh | 0.0223 (+12.6%) | 0.0199 (+40.1%) | 4.486 (−2.4%) | 3.632 (+18.7%) | 3.850 (−0.4%) |
| preliminary_euh | 0.0219 (+10.6%) | 0.0195 (+37.3%) | 4.424 (−3.7%) | 3.568 (+16.6%) | 4.041 (+4.6%) |
| **preliminary_mul_200k** | **0.0239 (+20.7%)** | **0.0207 (+45.8%)** | 4.510 (−1.8%) | 3.801 (+24.2%) | **4.245 (+9.8%)** |
| *o200k (200k ref)* | 0.0239 | 0.0176 | **4.786** | 3.519 | 4.040 (+4.5%) |

All four candidates compress the multilingual sets much more than Apertus v1,
and compress English a few percent less. Among the 131k candidates,
`preliminary_enh` has the smallest English loss (−2.4% vs Apertus). The 200k
candidate has the highest FLORES numbers. o200k (same vocabulary size)
compresses English more (4.786) and the full 205-language set much less (0.0176
vs 0.0207), so the larger vocabulary alone does not give multilingual breadth. On
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
[per_language_compression.png](per_language_compression.png): one panel per
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

Utilization here is the raw fraction of vocabulary ids used (used / vocab size).
The candidate table in DEVELOPMENT_RECORD.md excludes special and reserved tokens
from the denominator, so its Apertus v1 figure (0.561) sits slightly above the 0.556 shown
here; the candidate figures, with far fewer special tokens, agree to within 0.001.

Per-language vocabulary utilization (the raw count of distinct vocabulary ids
used to encode each language's 997-sentence FLORES corpus, not a fraction) is
plotted in
[per_language_vocab_utilization.png](per_language_vocab_utilization.png), one
panel per tokenizer over the same 15 languages as the compression plot. On the
low-resource tail the candidates use more distinct ids than Apertus v1, which
means dedicated subword tokens rather than byte fallback: Tibetan 525 to 1300
across the candidates against 230 for Apertus v1, and Tamil 853 to 1697 against
771. `preliminary_mul_200k` uses the most distinct ids on Arabic (6077) and
Hindi (2394) of the candidates, from its larger vocabulary.

### 3.1 Vocabulary-usage breakdown and scaffold tokens

Each merge-created token (the 256-byte base alphabet and the 124 special tokens
excluded) is run over a fixed corpus: a 49.2 MB FineWeb sample drawn over the
FLORES-200 language set at an equal byte budget per language (214,285 bytes,
seed 0; FineWeb2 per language, English from FineWeb-1, which is where English
lives; 210 of 210 languages returned text, 6 of them under budget), plus
FineMath-4+ and StarCoder (Python and JavaScript). Two per-token
rates are measured. The standalone rate is how often the token is emitted as a
final token, per million final tokens on the corpus. The survival rate is how
often it is emitted as a final token divided by how often it is built at all
(built = emitted as a final token, plus formed as an intermediate step and then
merged into something larger).

The four buckets partition the merge vocabulary (they sum to 100%): Active
(standalone rate at or above 5 per million), Rare (1 to 5 per million), Uncommon
(above 0 but below 1 per million), and Unseen (built zero times on this corpus,
in any role). Scaffold is an overlay on the Rare and Uncommon tokens, not a fifth
bucket: a token counts as scaffold if it is emitted on its own in fewer than 10%
of the times it is built (survival below 0.1). A scaffold token is one the
tokenizer builds mostly as an intermediate step toward a larger token, not as an
output on its own. In `preliminary_enh`, for example, `ould` is emitted alone 9
times but built 7,253 times, almost always inside ` would`. Scaffold tokens are
structurally needed to build the tokens that do surface; they are not removable
waste (distinct from the junk and dead-vocabulary metrics).

<!-- BEGIN TABLE: vocab-usage -->
| Tokenizer | Merge tokens | Active % | Rare % | Uncommon % | Unseen % | Scaffold % | Scaffold count |
|---|---|---|---|---|---|---|---|
| Apertus v1 | 129,816 | 13.62 | 26.54 | 52.53 | 7.30 | 3.49 | 4,531 |
| preliminary_mul | 130,692 | 15.38 | 31.21 | 50.21 | 3.19 | 2.80 | 3,659 |
| preliminary_enh | 130,692 | 16.30 | 31.89 | 47.23 | 4.58 | 2.39 | 3,124 |
| preliminary_euh | 130,692 | 16.26 | 28.73 | 49.13 | 5.88 | 2.40 | 3,137 |
| preliminary_mul_200k | 199,684 | 10.33 | 22.73 | 57.76 | 9.17 | 3.07 | 6,130 |
| *o200k (200k ref)* | 199,742 | 10.34 | 24.74 | 53.99 | 10.94 | 2.53 | 5,053 |
<!-- END TABLE: vocab-usage -->

At 131k the three candidates have a lower scaffold share than Apertus v1 (2.39 to
2.80% against 3.49%, or 3,124 to 3,659 tokens against 4,531). Their Active shares
are higher (15.38 to 16.30% against 13.62%) and their Unseen shares are lower
(3.19 to 5.88% against 7.30%): more of their vocabulary is exercised by this
corpus. `preliminary_mul_200k` has a 3.07% scaffold share, close to the 131k
candidates in percentage but 6,130 tokens in absolute count because the
vocabulary is 53% larger. Its Unseen share (9.17%) is higher than any 131k
tokenizer, which is the expected size effect: with 200k slots and a fixed corpus
a larger fraction of the vocabulary goes unused. Against the size-matched
reference it is still the lower of the two (o200k, also about 200k, has 10.94%
Unseen and a 2.53% scaffold share). Byte-fragment tokens are a small part of
every scaffold count (0.27 to 0.57% of the merge vocabulary across the six
tokenizers here).

## 4. Code-structure metrics

AST full-alignment is the fraction of AST-node spans whose token boundaries
match on both ends across the StarCoder sample. Operator isolation is the
fraction of operators emitted as standalone tokens; the column below is measured
on natural-language text (the 13 core languages), where the clean regex's
punctuation arm fires on nearly every operator.

| Tokenizer | AST full-alignment ↑ | Operator isolation (prose) ↑ |
|---|---|---|
| Apertus v1 | 0.488 | 0.373 |
| preliminary_mul | **0.689** | **0.991** |
| preliminary_enh | 0.679 | 0.990 |
| preliminary_euh | 0.682 | 0.990 |
| preliminary_mul_200k | 0.681 | 0.990 |
| *o200k (200k ref)* | 0.463 | 0.354 |

The candidates align to AST boundaries more often than Apertus v1 and o200k
(0.68 against 0.46 to 0.49), on the real StarCoder sample. The operator-isolation
column separates them too, but that separation is a property of prose: measured
on code, where operators are usually space-delimited, the same metric puts the
families much closer (Apertus 0.40, candidates 0.51 to 0.52, a gap of about 0.12
against the 0.62 prose gap). So the code-structure claim that holds here is the
AST alignment; operator isolation reflects the same regex difference (the clean
regex splits punctuation off adjacent characters) but is not by itself a
code-corpus measurement. One caution on the metric: isolation can be pushed to
1.0 by shattering compound operators, which a pure-punctuation regex does
(splitting `>=` into `>` `=`), so higher is not always better.

## 5. Encode throughput

Single-core encode throughput on the English FineWeb-Edu snippet (the same
1000-document snippet as the compression table in §1), measured through the
Hugging Face `tokenizers` Rust backend (`encode_batch`,
`add_special_tokens=False`, `RAYON_NUM_THREADS=1`), reported as the minimum over
11 timed repeats after warmup. Throughput is input bytes divided by encode time,
with MB = 10^6 bytes of input UTF-8 text. Numbers are single-core on a shared
login node, so absolute values vary roughly 10% run to run; treat
differences of a few percent as noise rather than a tokenizer effect.

<!-- BEGIN TABLE: throughput -->
| Tokenizer | Vocab | Ships `ignore_merges` | Encode MB/s |
|---|---|---|---|
| Apertus v1 | 131072 | true | 3.66 |
| preliminary_mul | 131072 | true | 3.25 |
| preliminary_enh | 131072 | true | 3.34 |
| preliminary_euh | 131072 | true | 3.32 |
| preliminary_mul_200k | 200064 | true | 3.34 |
| *o200k (200k ref)* | 200000 | **false** | 3.05 |
<!-- END TABLE: throughput -->

All four candidates now ship `ignore_merges=true` (they were built with it off,
like the rest of the training-library default; Apertus v1 already had it on).
The flag changes encode speed only, not tokenization: on this build it produced
identical token ids across all 211 FLORES languages, code, and Unicode edge
cases (emoji sequences, ligatures, combining marks, fullwidth and zero-width
characters).

`preliminary_mul_200k` was the last one still shipping with the flag off, and it
was turned on 2026-07-19. Its encode throughput went from 3.03 to 3.34 MB/s, a
1.10x gain, which brings it level with the 131k candidates instead of last. The
previous file is kept alongside it as
`tokenizer.json.bak_ignore_merges_false_2026-07-19`.

The candidates encode at 3.25 to 3.34 MB/s against Apertus v1's 3.66. The
remaining difference from Apertus v1 is the pretokenizer regex, which does more
work per byte. Vocabulary size is not the driver: the size-matched o200k
reference encodes at 3.05 MB/s, slower than every candidate, though it is the one
tokenizer here still without the flag so its figure is not directly comparable.
These are single-core numbers on a shared node: the candidates' spread (3.25 to
3.34) is within the run-to-run variance quoted above and should not be read as an
ordering between them.

## 6. Extrinsic: 1B-parameter LM

Each candidate's LM was trained on the same learned merges as its shipped
tokenizer, so the numbers below apply to the shipped tokenizer's tokenization of
real text. On exact identity: `preliminary_mul` is byte-identical to its shipped
file; `preliminary_enh` and `preliminary_euh` differ from theirs only in
special-token metadata (the merges are identical); `preliminary_mul_200k` used the
vocab-200000 predecessor of the shipped 200064-token build (ids 0 to 199999
identical, 64 merges appended in the shipped build). `preliminary_mul_200k`'s
larger vocabulary also makes its downstream numbers not a like-for-like comparison
with the 131k rows: a larger vocabulary changes the parameter count (embedding and
output tables) and the compute-optimal training-token budget. Its recipe is
otherwise the same family as `preliminary_euh` (same `plus2` pretokenizer and
Fr/De-boosted `consv2` data). The full per-tokenizer panel is in DEVELOPMENT_RECORD.md.

Protocol: nanochat GPT, depth-24 (~1B parameters), muP. Two training regimes per
tokenizer: a standard multilingual mix to about 10B tokens, and a from-scratch 20B
math+code mix (`mathcode-scratch`). Evaluations: validation BPB, downstream LM
FLORES BPB on the 31 training languages (BPB is normalized per byte, so it
compares across tokenizers), Code BPB, BLiMP, MultiBLiMP,
Belebele (31 languages), MGSM, GSM8K, HumanEval (0-shot), MBPP (3-shot), and
MC-math (k=5, 500 examples per dataset). One seed per tokenizer. Metric
definitions are in DEVELOPMENT_RECORD.md.

Validation BPB, FLORES BPB, Code BPB, BLiMP, MultiBLiMP, MGSM, and Belebele are
from the standard 10B run; MC-math, GSM8K, HumanEval, and MBPP are from the 20B
math+code run (`mathcode-scratch`). This is the same metric set and the same
values as the full per-tokenizer panel in DEVELOPMENT_RECORD.md (which also has
every tokenizer and the 10B-vs-20B justification). All values are read directly
from the run outputs.

| Metric | `preliminary_enh` | `preliminary_euh` | `preliminary_mul` | `preliminary_mul_200k` |
|---|---|---|---|---|
| Validation BPB ↓ | 0.725 | 0.725 | 0.728 | 0.720 |
| FLORES BPB (31 trained lang) ↓ | 1.164 | 1.167 | 1.167 | 1.163 |
| Code BPB ↓ | 0.529 | 0.532 | 0.531 | 0.524 |
| BLiMP acc ↑ | 0.820 | 0.820 | 0.814 | 0.821 |
| MultiBLiMP acc ↑ | 0.911 | 0.915 | 0.919 | 0.917 |
| MGSM ↑ | 0.016 | 0.011 | 0.014 | 0.010 |
| Belebele acc ↑ | 0.240 | 0.256 | 0.249 | 0.263 |
| MC-math ↑ | 0.273 | 0.279 | 0.285 | 0.247 |
| GSM8K flex (full 1319-item set) ↑ | 0.235 | 0.229 | 0.226 | 0.244 |
| HumanEval pass@1 ↑ | 0.146 | 0.165 | 0.159 | 0.171 |
| MBPP pass@1 ↑ [95% CI] | 0.224 [0.188, 0.260] | 0.182 [0.148, 0.216] | 0.212 [0.176, 0.248] | 0.228 [0.192, 0.264] |

The four candidates are close on general modeling: validation BPB 0.720 to 0.728,
FLORES BPB 1.163 to 1.167, and BLiMP 0.814 to 0.821. GSM8K, HumanEval, and MBPP
are single-seed; GSM8K and HumanEval have no computed CI (small samples, n=164
for HumanEval) and a paired-bootstrap check across all four plus Apertus v1
finds no significant pairwise difference on either. MBPP has a paired-bootstrap
95% CI (`bootstrap_mathcode_significance.py`, `generation_spec` v2-2026-07-30):
`preliminary_mul_200k` is highest (0.228 [0.192, 0.264]) and significantly ahead
of `preliminary_euh` (p_BH = 0.039), but not distinguishable from
`preliminary_enh` or `preliminary_mul` (p_BH = 0.889 and 0.504). `preliminary_mul`
is highest on MC-math (0.285, single run, no CI). At this model size and token
budget there is no consistent downstream separation strong enough to override
the intrinsic compression profile, which is what the recommendation rests on.

## Takeaways

- **`preliminary_mul_200k`** (recommended): with 53% more vocabulary it
  compresses English about as much as the others (FineWeb-Edu 4.510 bytes/token)
  and has the highest EU compression (EU-avg 4.129, German 4.363, French 4.295
  chars/token) and the smallest worst-language penalty (3.61x), while keeping the
  low-resource tail (Indic 2.759, Mandarin 1.149, Tibetan 2.518 chars/token). It
  compresses both the high-resource and low-resource languages more than the 131k
  candidates do. The larger vocabulary means a larger embedding and output table (200064 vs 131072).
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
  `preliminary_mul`. EU languages are compressed less than under Apertus v1.
- **`preliminary_euh`** (131k): highest EU compression of the 131k candidates
  (EU-avg 3.937; German +2.2%, French about the same at −2.2%). It allocates less vocabulary to
  Chinese, so Mandarin drops to 0.917, **below Apertus v1's 1.108 (−17%)**. By Gini it is
  the least fair of the four candidates. Appropriate if European compression is
  the priority and Chinese can be deprioritized.
