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
| `preliminary_mul` | 131017 | balanced multilingual (consv2; no EU boost) | `consv2` plus3 |
| `preliminary_enh` | 131072 | English-preserving (English-boosted, moderate EU) | `engfull_eu3` |
| `preliminary_euh` | 131072 | EU-dense, Fr/De-boosted (Chinese cut) | `frde2` |
| `preliminary_mul_200k` | **200000** | 200k all-rounder (head + tail, no 131k trade-off) | `eusino_v2c` |

## Primary recommendation

I recommend **`preliminary_euh`** as the primary candidate, given Apertus's
focus on Switzerland and the EU. It is the EU-dense option: French and German
improve notably (French 4.200, German 4.332 chars/token) and the European set
overall is denser than Apertus v1 (EU-avg 3.937 vs 3.785), at Apertus's own 131k
vocabulary size. Switzerland's national languages (German, French, Italian) and
the broader EU are where vocabulary is re-allocated to.

The cost is deliberat: EU tokens are funded by
cutting Sino-Tibetan, so `preliminary_euh` encodes Mandarin worse than Apertus v1 (−17%), 
and it is the least fair of the four candidates (still better than Apertus) across
all languages (Gini 0.138 / 0.122 on FLORES60 / FLORES200). It is the
recommended default when European coverage is the priority and Chinese 
can reasonably be deprioritized.

The other three are stronger on their own axes:

- **`preliminary_mul`** is the fairest (Gini 0.087 / 0.097) and most balanced —
  best Indic, best Mandarin (1.329) and best low-resource-tail compression of the
  131k set, with the fewest junk tokens (13). The better fit when broad
  multilingual fairness, rather than European density, is the goal.
- **`preliminary_enh`** keeps the most English of the 131k candidates
  (FineWeb-Edu 4.486 b/t, −2.4% vs Apertus) while retaining most of the
  multilingual and fairness gains. The better fit when English compression is the
  priority. German, French and other EU languages are worse off than in Apertus
  v1 though
- **`preliminary_mul_200k`** removes the head-vs-tail trade-off the 131k
  candidates face — it leads on EU, Indic, the FLORES sets and the worst-language
  factor (3.53×) while holding English near the others — but only by spending a
  200k vocabulary. The better fit if the larger embedding/output table (and the
  departure from Apertus's 131k) is acceptable.

**`preliminary_mul_200k` is a 200k-vocabulary tokenizer — a 53% step up from the
131k of the other three (and of Apertus v1).** A larger vocabulary compresses
more across the board purely from size, so its leads over the 131k candidates
are not apples-to-apples: the fair size-matched comparison for it is OpenAI's
o200k, also 200k. Read its advantages over the 131k rows as "what an extra 69k
vocabulary slots buy," not as a better data scheme. Its lower vocabulary-
utilization figures are the same size artifact (more slots, so a smaller used
fraction), not a defect.

No language model has been trained on any of these four tokenizers yet, so the
extrinsic section (§5) is pending for all of them.

Compression cells in §1 show `(% diff vs Apertus v1)`. Higher sent/tok and
higher bytes/token are better.

## 1. Compression — four corpora

`sent/tok` is FLORES sentences (lines) per token (higher = more compressed).
`b/t` is bytes per token (higher = more compressed). FineWeb2-proportional is a
7.5 MB multilingual sample (seed 0) drawn across 22 families with per-family
shares matching the natural FineWeb2 byte distribution; all tokenizers measured
on the identical sample. `FLORES EU b/t` is bytes/token over the ten EU languages
(deu/fra/spa/ita/por/nld/swe/pol/ron/dan) on FLORES.

| Tokenizer | FLORES60 sent/tok ↑ | FLORES200 sent/tok ↑ | FineWeb-Edu English b/t ↑ | FineWeb2-proportional b/t ↑ | FLORES EU b/t ↑ |
|---|---|---|---|---|---|
| Apertus v1 | 0.0198 | 0.0142 | 4.595 | 3.077 | 3.865 |
| preliminary_mul | 0.0234 (+18.2%) | 0.0204 (+43.7%) | 4.333 (−5.7%) | 3.771 (+22.6%) | 3.693 (−4.4%) |
| preliminary_enh | 0.0223 (+12.6%) | 0.0199 (+40.1%) | 4.486 (−2.4%) | 3.621 (+17.7%) | 3.850 (−0.4%) |
| preliminary_euh | 0.0219 (+10.6%) | 0.0195 (+37.3%) | 4.424 (−3.7%) | 3.559 (+15.7%) | 4.041 (+4.6%) |
| **preliminary_mul_200k** | **0.0240 (+21.2%)** | **0.0208 (+46.5%)** | 4.508 (−1.9%) | **3.783 (+22.9%)** | **4.160 (+7.6%)** |
| *o200k (200k ref)* | 0.0239 | 0.0176 | **4.786** | 3.533 | 4.040 (+4.5%) |

All four candidates compress the multilingual sets far more than Apertus v1 and
trade a few percent of English compression for it. `preliminary_enh` closes the
English gap most among the 131k candidates (−2.4% vs Apertus). The 200k
candidate leads the FLORES sets, but o200k (same vocab) is denser still on
English (4.786) while much weaker on the full 205-language set (0.0176 vs
0.0208) — the 200k vocabulary on its own does not buy multilingual breadth. On
the EU set, `preliminary_euh` (+4.6%) and `preliminary_mul_200k` (+7.6%) beat
Apertus v1, with `euh` the densest 131k EU option.

## 2. Fairness — Gini coefficient and worst-language factor

Worst-language factor is the multiplicative token-count increase, on the same
parallel FLORES content, between the worst-served language and English.

| Tokenizer | FLORES60 Gini ↓ | FLORES200 Gini ↓ | Worst FLORES200 factor ↓ |
|---|---|---|---|
| Apertus v1 | 0.205 | 0.313 | 14.70× (khm_Khmr) |
| preliminary_mul | **0.087** | **0.097** | 3.63× (sat_Olck) |
| preliminary_enh | 0.121 | 0.114 | 4.46× (sat_Olck) |
| preliminary_euh | 0.138 | 0.122 | 4.67× (sat_Olck) |
| preliminary_mul_200k | 0.113 | 0.112 | **3.53× (sat_Olck)** |
| *o200k (200k ref)* | 0.103 | 0.237 | 13.70× (sat_Olck) |


## 3. Vocabulary utilization and junk tokens

| Tokenizer | FLORES60 vocab util ↑ | FLORES200 vocab util ↑ | Junk tokens (≥8-char decorative runs) ↓ |
|---|---|---|---|
| Apertus v1 | 0.556 | 0.643 | 46 |
| preliminary_mul | 0.614 | **0.836** | **13** |
| preliminary_enh | 0.598 | 0.773 | 17 |
| preliminary_euh | 0.620 | 0.775 | 17 |
| preliminary_mul_200k | 0.543 | 0.734 | 17 |
| *o200k (200k ref)* | 0.475 | 0.590 | 255 |

The 200k candidate's lower utilization (0.543 / 0.734) is the vocabulary-size
artifact noted above, not waste — it uses more tokens in absolute terms.
o200k has 255 decorative-run/glitch tokens against 13–17 for the candidates.

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
| preliminary_mul_200k | 0.680 | 0.990 |
| *o200k (200k ref)* | 0.463 | 0.354 |

All four candidates respect code structure far better than both Apertus v1 and
o200k, which glue operators into surrounding tokens (0.35–0.37) and align to AST
boundaries less often.

## 5. Extrinsic — 1B-parameter LM

Pending. No language model has been trained on any of the four candidates yet,
so downstream BPB, BLiMP, MGSM, MC-math, GSM8K, HumanEval, and MBPP are not
available. This section will be filled once `-mathcode-scratch` and balanced LM
runs exist for the chosen candidate.

## Language character (FLORES chars/token, content-only; higher = denser)

This is some language/region specific compression numbers.

| Tokenizer | English | EU-avg | French | German | Italian | Indic | Mandarin | Tibetan |
|---|---|---|---|---|---|---|---|---|
| Apertus v1 | 4.742 | 3.785 | 4.296 | 4.238 | 4.024 | 2.352 | 1.108 | 0.433 |
| preliminary_mul | 4.555 | 3.596 | 3.584 | 3.617 | 3.683 | **2.775** | **1.329** | **2.941** |
| preliminary_enh | 4.715 | 3.748 | 3.771 | 3.784 | 3.899 | 2.427 | 1.134 | 2.489 |
| preliminary_euh | 4.650 | 3.937 | **4.200** | **4.332** | 4.034 | 2.328 | 0.917 | 2.222 |
| preliminary_mul_200k | 4.736 | **4.050** | 4.055 | 4.127 | **4.243** | 2.802 | 1.155 | 2.533 |

EU-avg = deu/fra/spa/ita/por/nld/swe/pol/ron/dan; French/German/Italian
(Switzerland's national languages) are broken out. `preliminary_euh` is the only
131k candidate to beat Apertus v1 on the EU average (3.937 vs 3.785), German
(4.332 vs 4.238) and Italian (4.034 vs 4.024), with French (4.200) just behind
Apertus (4.296). Every candidate encodes the low-resource tail far more densely
than Apertus v1 (Tibetan 2.2–2.9 vs 0.433).

## Takeaways

- **`preliminary_mul`** — the balanced multilingual choice. Fairest (Gini 0.087 /
  0.097), fewest junk tokens (13), best Indic/Mandarin/Tibetan of the 131k set,
  and best worst-language factor among the 131k candidates. It compresses English
  least (4.333 b/t, −5.7% worse than Apertus). 
- **`preliminary_enh`** — English focus. Best English compression of the 131k
  candidates (4.486 b/t, −2.4% vs Apertus) while keeping most of the multilingual
  and fairness gains; Indic and Chinese are reduced relative to `preliminary_mul`.
  This comes at a sacrifice to EU languages, which are worse compressed than under
  Apertus.
- **`preliminary_euh`** — EU-dense at 131k. Best EU of the 131k candidates
  (3.937; French/German much improved) but  Chinese is deprioritized, pushing Mandarin
  to 0.917 — **worse than Apertus v1's 1.108 (−17%)**. In terms of Gini,
  it is the least fair of the four candidate tokenizers. 
  Appropriate if European compression is a priority and Chinese can be deprioritzed.
- **`preliminary_mul_200k`** — With 53% more vocabulary it compresses English as much
  as the others (4.508) while increasing compression for the EU (4.050), Indic, the
  FLORES sets, and the worst-language factor (3.53×) — it removes the head-vs-tail
  trade-off the 131k candidates face. The cost is the larger embedding/output
  table and the departure from Apertus v1's 131k. Against the size-matched o200k
  it gives up ~6% English but is roughly 2× fairer across 205 languages, encodes
  the low-resource tail 2.5–3.7× more densely, and carries 17 junk tokens vs 255.

Notes: FineWeb2-proportional is a 7.5 MB sample (seed 0; all tokenizers on the
identical sample), so its Apertus value (3.077) differs slightly from earlier
6.3 MB runs (3.061). o200k has its own pretokenizer and no NFC normalizer
(inherent to the comparison). Extrinsic results are pending for all four
candidates.
