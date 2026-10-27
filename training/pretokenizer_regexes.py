# pretokenizer_regexes.py

"""
Pretokenization regex patterns from production tokenizers.

Each pattern is designed for use with HuggingFace tokenizers' Split pretokenizer
in "Isolated" mode, followed by a ByteLevel pretokenizer (use_regex=False).

Key design differences:
- GPT-4o: Case-insensitive contractions appended to word patterns, digits {1,3},
  Lu/Ll case-transition splitting. Dot attaches forward (.x, .append).
- GPT-4o (single-digit): Same as GPT-4o but \p{N} instead of \p{N}{1,3}.
  Improves arithmetic generalization (Dagan et al., 2024; Nogueira et al., 2021).
- GPT-4o (no contractions): Removes (?i:...) contraction suffixes.
  Tests multilingual impact of English-specific apostrophe handling.
- Mistral-Nemo: No contraction handling, single digits. Clean multilingual design.
  Apostrophes still attach forward via [^\r\n\p{L}\p{N}]? prefix.
- Qwen 3.5: Contractions as standalone first alternative, flat [\p{L}\p{M}]+
  word pattern (no CamelCase splitting), single digits, \p{M} excluded from
  symbol class.
- Clean-multi: Same word arms as Mistral-Nemo/Apertus (no contractions, Lu/Ll
  CamelCase split at lower->upper, single digits). Two tightenings vs
  Mistral-Nemo/Apertus:
    (1) leading-char allowance [^\r\n\p{L}\p{N}]? -> [ ]? (a literal space only);
    (2) the punctuation arm drops the trailing [\r\n/]*.
  Punctuation: never attaches forward to a word (only a leading space does) —
    don't -> ['don', "'", 't'], self.x -> ['self', '.', 'x'], a/b -> ['a','/','b']
    (Mistral-Nemo/Apertus attach it: ['don',"'t"], ['self','.x']). A punctuation
    run also does not absorb trailing newlines: *\n -> ['*', '\n'] (Apertus ['*\n']).
  Whitespace by type: a single leading SPACE attaches to a following word AND to a
    punctuation run in BOTH clean-multi and Apertus (the punctuation arm is " ?",
    space-prefixed, in both: ' *' -> [' *']). The TAB difference is confined to the
    WORD arm: Apertus's [^\r\n\p{L}\p{N}]? attaches a leading tab to a word
    (\tword -> ['\tword']) but clean-multi's [ ]? does not (\tword -> ['\t','word']);
    for a punctuation run a leading tab attaches in NEITHER ('\t*' -> ['\t','*'] in
    both). A NEWLINE never attaches forward in any variant. Surplus spaces split off
    (  word -> [' ',' word']) and newline runs are their own token (\s*[\r\n]+); the
    three whitespace arms are byte-identical to Apertus's, so the only clean-multi-
    vs-Apertus differences here are the word-arm tab-attachment and the punctuation
    tail (Apertus's [\r\n/]* absorbs a trailing slash as well as newlines).

Known shared behaviors:
- GPT-4o/Mistral-Nemo/Apertus: a single leading punctuation char (incl. the dot)
  attaches forward to the following word (self.x -> ['self', '.x']). Clean-multi
  does NOT — its [ ]? leading allowance is a space only, so punctuation is split
  off (self.x -> ['self', '.', 'x']).
- All: no script-boundary split (Python是语言 -> one pre-token)
- All: \s+(?!\S) backtracks, so a leading *space* attaches to the next word. A
  leading TAB attaches only for GPT-4o/Mistral-Nemo/Apertus (their [^\r\n\p{L}\p{N}]?
  includes tab); clean-multi's [ ]? leaves a tab as its own token.
- GPT-4o/Mistral-Nemo: CamelCase splits at lowercase->uppercase transition,
  but NOT within uppercase runs (HTMLParser stays whole)
- Qwen 3.5: no CamelCase splitting at all
"""

REGEX_GPT4O = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

REGEX_GPT4O_SINGLEDIGIT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

REGEX_GPT4O_NOCONTRACTIONS = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

REGEX_MISTRAL_NEMO = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

REGEX_QWEN35 = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

REGEX_CLEAN_MULTI = (
    r"[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Variant of REGEX_CLEAN_MULTI with two targeted additions:
#   (1) The Tibetan tsek U+0F0B is added to the word-arm leading allowance
#       ([ ]? -> [ \x{0F0B}]?). The tsek is the canonical syllable separator in
#       Tibetan-script languages — under plain clean-multi every tsek became its
#       own pretoken, fragmenting bod_Tibt/dzo_Tibt 1.5–1.7x vs Apertus on FLORES.
#       With the tsek attached forward to the next syllable, Tibetan tokenises in
#       line with Apertus on those languages while leaving every other script
#       unchanged (tsek does not occur outside Tibetan-script text).
#   (2) The seven canonical English contraction suffixes are added as a
#       standalone branch placed first in the alternation, so they match before
#       the punctuation arm — don't -> [don, 't] instead of [don, ', t]. Unlike
#       GPT-4o's appended-suffix form this still keeps the apostrophe out of the
#       preceding word token (consistent with clean-multi's "punctuation never
#       attaches to a word" rule); the contraction simply tokenises as one unit.
# All other arms (numbers, punctuation, whitespace) are byte-identical to
# REGEX_CLEAN_MULTI. No other script-specific separator is given the same
# treatment as the tsek — see the agent review notes for which scripts (if any)
# might warrant analogous additions.
REGEX_CLEAN_MULTI_PLUS = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[ \x{0F0B}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ \x{0F0B}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# ---------------------------------------------------------------------------
# Run-capped variants
# ---------------------------------------------------------------------------
# Each *_CAPPED regex mirrors its base EXACTLY except that runs of punctuation/
# symbols and whitespace are length-bounded, so BPE cannot merge a long
# decorative run (e.g. '----...', '****...', '====...', '////...', long space
# runs) into a single junk vocabulary token. Verified to produce byte-identical
# pre-tokenization to the base on normal text/code/math/multilingual input
# (only pathological long runs differ) and to conserve all characters (no
# whitespace-fidelity loss).
#
# Caps are kept inline to match the file's existing literal style (cf. the
# digit cap \p{N}{1,3}):
#   - punctuation/symbol runs -> {1,16}  (16 preserves path '../../' and LaTeX
#     markup, which a tighter cap like {1,3} would fragment)
#   - space/tab and trailing-whitespace runs -> {1,16}
#   - the trailing '/' is dropped from the punctuation arm's [\r\n/]* tail
#     (slashes are already in the main class) so long '/' runs are capped too
#   - newline runs use \s{0,16}[\r\n]{1,16}: behaviour-identical to the base
#     \s*[\r\n]+ on real text; pure newline runs cap at <=32 (pair with a
#     trainer max_token_length backstop if a hard <=16 cap is required)
# Digit caps are inherited unchanged from the base regex.

REGEX_GPT4O_CAPPED = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]{1,16}[\r\n]{0,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

REGEX_CLEAN_MULTI_CAPPED = (
    r"[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ ]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# Run-capped variant of REGEX_CLEAN_MULTI_PLUS. Same two additions vs the base
# capped regex (tsek attaches forward; English contractions as a standalone
# first branch), plus the standard {1,16} caps on punctuation/whitespace runs.
REGEX_CLEAN_MULTI_PLUS_CAPPED = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[ \x{0F0B}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ \x{0F0B}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# Further extension: also attach a leading ASCII apostrophe (U+0027) and right
# single-quotation mark (U+2019) to the next word. Motivated by Apertus's
# advantage on French/Italian/Catalan/Maltese elision (l'eau, c'est, l'arte)
# where the apostrophe is a per-word morpheme boundary, not a sentence punct.
# The English-contractions branch from `_PLUS_CAPPED` is intentionally dropped
# here: with the apostrophe in the leading-char class, all seven canonical
# contractions tokenize identically through the word arms (don't -> [don, 't],
# I've -> [I, 've], etc.), and the branch was actively *splitting* edge cases
# like 'tis -> ['t, is] and 'sword -> ['s, word] that are cleanly one token
# without it. Caveats / harms: asymmetric quoted-string tokenization in code
# (opening ' glues to the word, closing ' does not); right-curly U+2019
# attaches but left-curly U+2018 does not (asymmetric pair handling); see the
# accompanying analysis.
REGEX_CLEAN_MULTI_PLUS2_CAPPED = (
    r"[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# Further extension: allow ASCII apostrophe / right curly quote to TRAIL a word
# as well (in addition to leading attach), guarded by (?!\p{L}) so the trailing
# match never steals from a following English contraction or French elision.
# Targets the residual Maltese-morpheme gap (ta', gh', m'), English dialect
# (talkin'), and math prime/derivative notation (f', A'). Multilingual base
# (don't, Bob's, l'eau, c'est, we’re, &'a Rust lifetime, Pinyin Xi'an, Wolof
# glottal stop) is preserved by the guard. Documented harms: paired single-
# quoted code/dialogue strings lose their "closing-quote as separable boundary"
# property (closing ' glues to the last inner word); curly-quote pair becomes
# asymmetric in a different way than plus2 (right ’ glues to body, left ‘
# standalone). See accompanying analysis.
REGEX_CLEAN_MULTI_PLUS3_CAPPED = (
    r"[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:[\x{0027}\x{2019}](?!\p{L}))?"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:[\x{0027}\x{2019}](?!\p{L}))?"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# plus3 with a single-character repeat-run cap of 8 in both word and punct
# arms. Implementation: a guard arm at the top of the regex matches any
# non-whitespace, non-digit character followed by 7 identical copies (= 8 same
# chars), optionally preceded by the same leading-char class used by the word
# arms (space, tibetan tsek, ASCII or curly apostrophe). Because regex
# alternation is leftmost-first in Oniguruma, the guard fires before the
# word/punct arms whenever a same-character run of 8+ is encountered, splitting
# the run into 8-char chunks. Mixed-character punctuation runs are unaffected
# (still up to 16 chars via the punct arm). Whitespace AND digits are excluded
# from the guard: whitespace so the existing whitespace arms keep handling
# space/newline runs, and digits (\p{N}) so the single-digit arm always applies
# to every digit. Excluding digits was added after an earlier guard of [^\s]
# was found to capture 8+ identical-digit runs (e.g. '00000000', '11111111'),
# which broke the single-digit invariant and let multi-digit tokens form.
# Motivation: prevent chrome tokens like 'FFFFFFFFFFFFFFFF', 'OOOOOOOOOOOOOOOO',
# 'NNNNNNNNNNNNNNNN', '----------------', '================' etc. from
# being aggregated into single 16-char vocab tokens.
REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED = (
    r"[ \x{0F0B}\x{0027}\x{2019}]?(?<g>[^\s\p{N}])\k<g>{7}"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:[\x{0027}\x{2019}](?!\p{L}))?"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:[\x{0027}\x{2019}](?!\p{L}))?"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# plus2 + the same repcap8 guard arm as plus3_repcap8 (guard-only): caps
# standalone 8+ same-character runs at 8. Identical to
# REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED except the two word arms omit the
# trailing apostrophe group `(?:[\x{0027}\x{2019}](?!\p{L}))?` (the only
# plus2-vs-plus3 difference). Built for the plus2-vs-plus3 ablation.
REGEX_CLEAN_MULTI_PLUS2_REPCAP8_CAPPED = (
    r"[ \x{0F0B}\x{0027}\x{2019}]?(?<g>[^\s\p{N}])\k<g>{7}"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[ \x{0F0B}\x{0027}\x{2019}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# Apertus (swiss-ai/Apertus-70B-2509) uses the Mistral-Nemo pretokenization
# scheme verbatim (verified by extracting its tokenizer.json, 2026-05-19).
REGEX_APERTUS = REGEX_MISTRAL_NEMO

REGEX_APERTUS_CAPPED = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]{1,16}[\r\n]{0,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
)

# Convenience mapping for iteration/lookup
ALL_REGEXES = {
    "gpt4o": REGEX_GPT4O,
    "gpt4o-singledigit": REGEX_GPT4O_SINGLEDIGIT,
    "gpt4o-nocontractions": REGEX_GPT4O_NOCONTRACTIONS,
    "mistral-nemo": REGEX_MISTRAL_NEMO,
    "qwen35": REGEX_QWEN35,
    "clean-multi": REGEX_CLEAN_MULTI,
    "clean-multi-plus": REGEX_CLEAN_MULTI_PLUS,
    "apertus": REGEX_APERTUS,
    "gpt4o-capped": REGEX_GPT4O_CAPPED,
    "clean-multi-capped": REGEX_CLEAN_MULTI_CAPPED,
    "clean-multi-plus-capped": REGEX_CLEAN_MULTI_PLUS_CAPPED,
    "clean-multi-plus2-capped": REGEX_CLEAN_MULTI_PLUS2_CAPPED,
    "clean-multi-plus3-capped": REGEX_CLEAN_MULTI_PLUS3_CAPPED,
    "clean-multi-plus2-repcap8-capped": REGEX_CLEAN_MULTI_PLUS2_REPCAP8_CAPPED,
    "clean-multi-plus3-repcap8-capped": REGEX_CLEAN_MULTI_PLUS3_REPCAP8_CAPPED,
    "apertus-capped": REGEX_APERTUS_CAPPED,
}
