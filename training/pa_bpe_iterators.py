#!/usr/bin/env python3
"""Shared iterator library for PA-BPE 30-language training.

Used by:
  * train_tokenizer.py — feeds per-language iterators directly into
    ParityBpeTrainer.train_from_iterator (or chains them for the BPE baseline)
  * validate_languages.py — pulls small samples for fasttext/langid sanity

Provides two main classes:
  * LanguageTrainCorpus — subsampling parquet stream bounded by a byte quota,
    with per-shard caps, deterministic seed, and shard_log recording.
  * LanguageDevCorpus — line-by-line iterator over a FLORES+ .txt file.

Both implement __iter__ so they can be passed directly to
``train_from_iterator`` as ``Iterable[str]``.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pyarrow.parquet as pq


# ----- constants ------------------------------------------------------------

# Shard-level concurrency cap:
#   max_bytes_per_shard = max(quota_bytes / min(K, n_shards), MIN_PER_SHARD_BYTES)
# K bounds the "maximum number of shards that fully share a quota" so that a
# language with many shards is forced to draw from at least K of them.
DEFAULT_SHARD_CAP_K = 16
MIN_PER_SHARD_BYTES = 8 * 1024 * 1024   # 8 MiB
FIRST_1_MIB = 1 * 1024 * 1024
READ_BATCH_SIZE = 4096


# ----- helpers --------------------------------------------------------------

def _sha256_first_1mib(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(FIRST_1_MIB))
    return h.hexdigest()


def _enumerate_parquet_shards(train_root: Path) -> List[Path]:
    """Recursively find .parquet files under train_root and return them
    sorted by path (for determinism before shuffling)."""
    shards = sorted(train_root.rglob("*.parquet"))
    return shards


# ----- shard log ------------------------------------------------------------

@dataclass
class ShardRecord:
    path: str
    size_bytes: int
    sha256_first_1mib: str
    rows_emitted: int
    bytes_emitted: int
    row_groups_read: int

    def to_dict(self):
        return self.__dict__.copy()


# ----- training corpus ------------------------------------------------------

@dataclass
class LanguageTrainCorpus:
    """Iterable yielding text records from a language's parquet shards up to
    ``quota_bytes`` of UTF-8 text.

    After iteration completes, ``shard_log`` holds one ShardRecord per shard
    that was actually touched, in the order they were consumed.
    """
    train_root: Path
    lang_code: str
    quota_bytes: int
    seed: int
    text_column: str = "text"
    shard_cap_k: int = DEFAULT_SHARD_CAP_K
    min_per_shard_bytes: int = MIN_PER_SHARD_BYTES
    verbose: bool = False

    # Populated during iteration.
    shard_log: List[ShardRecord] = field(default_factory=list)
    actual_bytes: int = 0
    actual_lines: int = 0

    def __post_init__(self):
        self.train_root = Path(self.train_root)
        if not self.train_root.exists():
            raise FileNotFoundError(f"train_root does not exist: {self.train_root}")

    def _iter_rows(self) -> Iterator[str]:
        # Reset per-iteration state — allow a single instance to be re-iterated
        # with the same seed and produce the same output.
        self.shard_log = []
        self.actual_bytes = 0
        self.actual_lines = 0

        shards = _enumerate_parquet_shards(self.train_root)
        if not shards:
            raise RuntimeError(
                f"no parquet shards under {self.train_root} for {self.lang_code}"
            )

        # Seed RNG deterministically per (global seed, language).
        rng = random.Random(f"{self.seed}:{self.lang_code}")
        shard_order = list(range(len(shards)))
        rng.shuffle(shard_order)

        max_bytes_per_shard = max(
            self.quota_bytes // min(self.shard_cap_k, len(shards)),
            self.min_per_shard_bytes,
        )

        for shard_idx in shard_order:
            if self.actual_bytes >= self.quota_bytes:
                break
            shard_path = shards[shard_idx]
            yield from self._iter_one_shard(shard_path, rng, max_bytes_per_shard)

    def _iter_one_shard(
        self,
        shard_path: Path,
        rng: random.Random,
        max_bytes_this_shard: int,
    ) -> Iterator[str]:
        try:
            pf = pq.ParquetFile(str(shard_path))
        except Exception as e:
            if self.verbose:
                print(f"[{self.lang_code}] SKIP unreadable shard {shard_path}: {e}")
            return
        n_rg = pf.num_row_groups
        if n_rg == 0:
            return

        # Random row-group visit order avoids intra-shard sort bias (parquets
        # are often sorted by crawl date / URL / dedup cluster).
        rg_order = list(range(n_rg))
        rng.shuffle(rg_order)

        rec = ShardRecord(
            path=str(shard_path),
            size_bytes=shard_path.stat().st_size,
            sha256_first_1mib=_sha256_first_1mib(shard_path),
            rows_emitted=0,
            bytes_emitted=0,
            row_groups_read=0,
        )

        for rg_idx in rg_order:
            if rec.bytes_emitted >= max_bytes_this_shard:
                break
            if self.actual_bytes >= self.quota_bytes:
                break
            try:
                tbl = pf.read_row_group(rg_idx, columns=[self.text_column])
            except Exception as e:
                if self.verbose:
                    print(f"[{self.lang_code}] SKIP row group {rg_idx} of "
                          f"{shard_path}: {e}")
                continue
            rec.row_groups_read += 1

            # Iterate column values without materializing the full table.
            col = tbl.column(self.text_column)
            for val in col:
                text = val.as_py()
                if not text:
                    continue
                byte_len = len(text.encode("utf-8"))
                rec.rows_emitted += 1
                rec.bytes_emitted += byte_len
                self.actual_lines += 1
                self.actual_bytes += byte_len
                yield text
                if rec.bytes_emitted >= max_bytes_this_shard:
                    break
                if self.actual_bytes >= self.quota_bytes:
                    break

        self.shard_log.append(rec)

    def __iter__(self) -> Iterator[str]:
        return self._iter_rows()


# ----- listed-file training corpus ------------------------------------------

@dataclass
class ListedFileCorpus:
    """Iterable yielding text records from an explicit list of parquet files.

    Unlike ``LanguageTrainCorpus`` which globs a directory and applies a byte
    quota, this reads every record from every listed file. Used for the
    grouped-config training pipeline where the config JSON already hand-picks
    which parquets belong to each language group.

    If the parquet schema has no column named ``text_column``, the iterator
    auto-detects using ``text_column_candidates`` (tries "text" first, then
    "content" for starcoder-style files).

    After iteration, ``shard_log`` holds one ShardRecord per file touched.
    """
    file_paths: List[Path]
    group_name: str
    text_column: str = "text"
    text_column_candidates: tuple = ("text", "content")
    quota_bytes: Optional[int] = None  # None = read everything
    verbose: bool = False

    shard_log: List[ShardRecord] = field(default_factory=list)
    actual_bytes: int = 0
    actual_lines: int = 0

    def __post_init__(self):
        self.file_paths = [Path(p) for p in self.file_paths]
        for p in self.file_paths:
            if not p.exists():
                raise FileNotFoundError(f"{p} (group {self.group_name})")

    def _file_row_gen(self, shard_path: Path):
        """Lazy ``(text, byte_len)`` generator for one parquet file.

        Returns ``None`` if the file is unreadable or has no text column —
        and prints a loud, ungated WARNING naming the excluded file (the
        project forbids silently dropping a configured artifact). The
        ``ShardRecord`` is appended eagerly at generator-creation time so a
        file that is configured-but-never-drawn (or drawn-but-empty) is still
        visible in ``shard_log`` / ``run_manifest.json`` with
        ``rows_emitted=0`` rather than vanishing.
        """
        try:
            pf = pq.ParquetFile(str(shard_path))
        except Exception as e:
            print(f"[{self.group_name}] COVERAGE WARNING: unreadable parquet, "
                  f"EXCLUDED from training: {shard_path}: {e}", flush=True)
            return None

        schema_cols = set(pf.schema_arrow.names)
        col = next((c for c in (self.text_column, *self.text_column_candidates)
                    if c in schema_cols), None)
        if col is None:
            print(f"[{self.group_name}] COVERAGE WARNING: no text column "
                  f"(tried {(self.text_column,) + self.text_column_candidates}), "
                  f"EXCLUDED from training: {shard_path}", flush=True)
            return None

        rec = ShardRecord(
            path=str(shard_path),
            size_bytes=shard_path.stat().st_size,
            sha256_first_1mib=_sha256_first_1mib(shard_path),
            rows_emitted=0,
            bytes_emitted=0,
            row_groups_read=0,
        )
        self.shard_log.append(rec)

        def gen():
            for rg_idx in range(pf.num_row_groups):
                try:
                    tbl = pf.read_row_group(rg_idx, columns=[col])
                except Exception as e:
                    print(f"[{self.group_name}] WARNING: unreadable row group "
                          f"{rg_idx} of {shard_path}: {e}", flush=True)
                    continue
                rec.row_groups_read += 1
                for val in tbl.column(col):
                    text = val.as_py()
                    if not text:
                        continue
                    blen = len(text.encode("utf-8"))
                    rec.rows_emitted += 1
                    rec.bytes_emitted += blen
                    yield text, blen
        return gen()

    def _log_coverage(self, n_input: int) -> None:
        """Loudly report any configured file that contributed nothing.

        Distinguishes (a) files with no ShardRecord at all (unreadable / no
        text column — already warned individually) from (b) files that have a
        record but emitted 0 rows. Both are exclusions from training and must
        be auditable from ``train.log`` per the no-silent-fallback rule.
        """
        n_rec = len(self.shard_log)
        n_norecord = n_input - n_rec
        zero = [r.path for r in self.shard_log if r.rows_emitted == 0]
        if n_norecord or zero:
            print(f"[{self.group_name}] COVERAGE SUMMARY: {n_input} configured "
                  f"files; {n_norecord} unreadable/no-text-col (no record); "
                  f"{len(zero)} readable but emitted 0 rows: "
                  f"{[Path(p).parent.name + '/' + Path(p).name for p in zero]}",
                  flush=True)

    def _iter_rows(self) -> Iterator[str]:
        self.shard_log = []
        self.actual_bytes = 0
        self.actual_lines = 0

        n_input = len(self.file_paths)
        gens = []
        for p in self.file_paths:
            g = self._file_row_gen(p)
            if g is not None:
                gens.append(g)
        if not gens:
            if n_input:
                print(f"[{self.group_name}] COVERAGE WARNING: 0 of {n_input} "
                      f"configured files were readable with a text column",
                      flush=True)
            return

        # No quota: read every file fully, in config order (legacy semantics).
        if self.quota_bytes is None:
            for g in gens:
                for text, blen in g:
                    self.actual_lines += 1
                    self.actual_bytes += blen
                    yield text
            self._log_coverage(n_input)
            return

        quota = self.quota_bytes
        n = len(gens)
        emitted = [0] * n
        active = [True] * n

        # --- Seed pass: one row per non-empty file, BEFORE any cap-limited
        # distribution. This makes the coverage guarantee *unconditional* —
        # every readable, non-empty file contributes at least one row even if
        # quota/n is smaller than a single row (the failure the FCFS bug and
        # the naive proportional cap both have). Bounded extra cost: at most
        # one row per file (for the moddata configs ≈ n·maxrow ≈ 87·0.8 MB ≪
        # the 1.2 GB family quota and ≪ the ~8 GB SLURM headroom).
        for i, g in enumerate(gens):
            try:
                text, blen = next(g)
            except StopIteration:
                active[i] = False
                continue
            emitted[i] += blen
            self.actual_lines += 1
            self.actual_bytes += blen
            yield text

        # --- Round-robin the remaining budget with a per-file cap that is
        # raised (redistributed) each pass by the leftover budget spread over
        # files that still have data. Memory bound: the inner guard
        # `self.actual_bytes < quota` is independent of `cap`, so total yield
        # never exceeds quota by more than one row per file visited in the
        # crossing pass (negligible vs headroom).
        cap = quota / n
        while self.actual_bytes < quota and any(active):
            progressed = False
            for i, g in enumerate(gens):
                if not active[i]:
                    continue
                while emitted[i] < cap and self.actual_bytes < quota:
                    try:
                        text, blen = next(g)
                    except StopIteration:
                        active[i] = False
                        break
                    emitted[i] += blen
                    self.actual_lines += 1
                    self.actual_bytes += blen
                    progressed = True
                    yield text
                if self.actual_bytes >= quota:
                    break
            if not progressed:
                break
            live = sum(active)
            if live and self.actual_bytes < quota:
                cap += (quota - self.actual_bytes) / live

        self._log_coverage(n_input)

    def __iter__(self) -> Iterator[str]:
        return self._iter_rows()


# ----- dev corpus -----------------------------------------------------------

@dataclass
class LanguageDevCorpus:
    """Iterable over non-empty lines of a FLORES+ dev file.

    If ``max_lines`` is set, yields at most that many non-empty lines (the
    first N in file order). Used to study the effect of dev-set size on the
    parity-aware selection signal.
    """
    flores_path: Path
    max_lines: Optional[int] = None
    actual_lines: int = 0
    actual_bytes: int = 0

    def __post_init__(self):
        self.flores_path = Path(self.flores_path)
        if not self.flores_path.exists():
            raise FileNotFoundError(self.flores_path)

    def __iter__(self) -> Iterator[str]:
        self.actual_lines = 0
        self.actual_bytes = 0
        with open(self.flores_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    if self.max_lines is not None and self.actual_lines >= self.max_lines:
                        break
                    self.actual_lines += 1
                    self.actual_bytes += len(line.encode("utf-8"))
                    yield line


# ----- quota helpers --------------------------------------------------------

def compute_quota_bytes(
    share: float,
    total_bytes: int,
    floor_bytes: int,
) -> int:
    """quota = max(floor, share * total). The floor handles low-resource
    languages where mC4's natural distribution would otherwise give too
    little data for 128k BPE to produce stable merges."""
    return int(max(floor_bytes, share * total_bytes))
