"""Tokenize exact root-aligned Data with a pinned local tokenizer asset.

Run with ``python -m tokenization_scripts.preprocess_megatron --help``.
No model download, dataset preparation or selection occurs. ADT tracking is opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from data_pipeline_pretrain.pipeline import root_data as rd
from datatrove.executor.local import LocalPipelineExecutor
from tokenizers import Tokenizer


@dataclass(frozen=True)
class TokenizationJob:
    """Exact input/output refs and local execution controls.

    Tasks are total native partitions; workers run tasks concurrently; threads
    bound each worker's Rayon pool. Batches count rows and UTF-8 bytes, and pair
    limits count sequences. An oversize row runs alone without truncation.
    ``tokenizer_revision`` is the full Git commit containing ``tokenizer_file``.
    Native logs and all settings must match when the operator requests resume.
    """

    data_root: Path
    root_ref: rd.ArtifactRef
    input_ref: rd.ArtifactRef
    output_ref: rd.ArtifactRef
    tokenizer_file: Path
    tokenizer_revision: str
    logging_dir: Path
    prefix_tokens: tuple[int, ...]
    suffix_tokens: tuple[int, ...]
    tasks: int = 1
    workers: int = 1
    threads: int = 1
    batch_size: int = 1000
    batch_bytes: int = 32 * 1024**2
    max_sequences_per_shard: int = 100000
    resume: bool = False
    proposal: rd.TrackedProposal | None = None

    def __post_init__(self) -> None:
        for ref in (self.root_ref, self.input_ref, self.output_ref):
            if ref.kind != "data":
                raise ValueError("root, input and output refs must identify Data")
        if self.output_ref in (self.root_ref, self.input_ref):
            raise ValueError("output must be a distinct aligned Data revision")
        for name in (
            "tasks",
            "workers",
            "threads",
            "batch_size",
            "batch_bytes",
            "max_sequences_per_shard",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.workers > self.tasks:
            raise ValueError("workers cannot exceed total tasks")
        if not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision):
            raise ValueError(
                "tokenizer revision must be a full 40-character Git commit"
            )
        if type(self.resume) is not bool:
            raise ValueError("resume must be a boolean")


def _git(folder: Path, *arguments: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(folder), *arguments], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"cannot identify local tokenizer asset: {error.stderr.decode().strip()}"
        ) from error


def _repository_url(origin: str) -> str:
    """Keep a fetchable source identity without URL credentials or Git rewrites."""
    scp = re.fullmatch(r"[^@/]+@([^:/]+):(.+)", origin)
    if scp:
        origin = f"ssh://{scp[1]}/{scp[2]}"
    parsed = urlsplit(origin)
    if parsed.scheme not in ("https", "http", "ssh", "git") or not parsed.hostname:
        raise ValueError(
            "tokenizer origin must be a fetchable HTTPS or SSH repository URL"
        )
    host = parsed.netloc.rsplit("@", 1)[-1]
    if parsed.hostname.lower() == "github.com":
        path = parsed.path.rstrip("/")
        if not path.endswith(".git"):
            path += ".git"
        return urlunsplit(("https", "github.com", path, "", ""))
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def load_asset(path: Path, revision: str) -> tuple[bytes, dict[str, Any]]:
    """Verify local asset bytes against an exact Git commit, without network I/O.

    The asset must be tracked beneath a Git worktree with an origin URL. A local
    edit, untracked asset or wrong revision fails before output mutation. Returning
    the verified bytes keeps worker loading independent of later filesystem edits.
    """
    path = path.resolve(strict=True)
    repository = Path(
        _git(path.parent, "rev-parse", "--show-toplevel").decode().strip()
    )
    relative = path.relative_to(repository).as_posix()
    if _git(repository, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("tokenizer revision must identify a Git commit")
    pinned = _git(repository, "show", f"{revision}:{relative}")
    if path.read_bytes() != pinned:
        raise ValueError(
            "local tokenizer bytes differ from the configured Git revision"
        )
    # get-url expands machine insteadOf rewrites, which may contain credentials.
    origin = _repository_url(
        _git(repository, "config", "--get", "remote.origin.url").decode().strip()
    )
    return pinned, {"repository": origin, "commit": revision, "path": relative}


@dataclass
class LocalBodyEncoder:
    """Load verified local JSON and encode bodies with special tokens disabled.

    Asset truncation and padding are explicitly disabled; DPP alone inserts the
    job's prefix and suffix. One output body corresponds to every input text.
    """

    asset: bytes

    @cached_property
    def tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer.from_str(self.asset.decode("utf-8"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    def __call__(self, texts: list[str]) -> list[list[int]]:
        return [
            row.ids
            for row in self.tokenizer.encode_batch(texts, add_special_tokens=False)
        ]


def run_job(
    job: TokenizationJob,
) -> tuple[rd.CollectionMetadata, rd.CollectionMetadata]:
    """Guard both outputs, run native tasks, qualify sequences and publish lengths.

    Raises on input/encoding/alignment/corruption errors; no row is filtered or
    rehydrated. ADT is loaded/contacted only for an explicit ``job.proposal``.
    Tracked mode checks the whole proposal before production and requires the
    coordinated ADT SDK on Python 3.11+. Resume permits mutation and reuses native
    completion records: it does not prove compatible inputs/config/logs or solve
    the separately held completion/lifecycle policy.
    """
    asset, identity = load_asset(job.tokenizer_file, job.tokenizer_revision)
    encoder = LocalBodyEncoder(asset)
    vocabulary_size = encoder.tokenizer.get_vocab_size(with_added_tokens=True)
    os.environ["RAYON_NUM_THREADS"] = str(job.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    encoding = {
        "tokenizer": identity,
        "add_special_tokens": False,
        "truncation": None,
        "padding": None,
    }

    def produce() -> tuple[rd.CollectionMetadata, rd.CollectionMetadata]:
        primary = rd.read_metadata(
            job.data_root, job.root_ref, job.input_ref
        ).semantics.get("primary_content")
        if not isinstance(primary, str) or not primary:
            raise ValueError("input Data must declare a primary text field")
        source = rd.RootParquetReader(
            job.data_root,
            job.root_ref,
            job.input_ref,
            text_key=primary,
            id_key=None,
            batch_size=job.batch_size,
        )
        writer = rd.TokenWriter(
            source,
            job.output_ref,
            tasks=job.tasks,
            vocabulary_size=vocabulary_size,
            encoding=encoding,
            encoder=encoder,
            prefix_tokens=job.prefix_tokens,
            suffix_tokens=job.suffix_tokens,
            batch_size=job.batch_size,
            batch_bytes=job.batch_bytes,
            max_sequences_per_shard=job.max_sequences_per_shard,
        )
        writer.prepare()
        executor = LocalPipelineExecutor(
            [source, writer],
            tasks=job.tasks,
            workers=job.workers,
            logging_dir=str(job.logging_dir),
            start_method="spawn",
        )
        executor.run()
        return writer.finalize(executor)

    return rd.run_producer(
        job.data_root,
        [
            rd.OutputCollection(job.root_ref, ref)
            for ref in (job.output_ref, rd.token_length_ref(job.output_ref))
        ],
        produce,
        resume=job.resume,
        proposal=job.proposal,
    )


def positive_int(value: str) -> int:
    """Argparse positive integer, used for counts and byte limits."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def token_ids(value: str) -> tuple[int, ...]:
    """Parse comma-separated boundary IDs; an empty string means no boundary."""
    if not value:
        return ()
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated nonnegative integer IDs"
        ) from error
    if any(token < 0 for token in result):
        raise argparse.ArgumentTypeError("boundary IDs must be nonnegative")
    return result


def get_args(argv: list[str] | None = None) -> TokenizationJob:
    """Parse the one supported root-aligned CLI; boundary IDs are explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="physical local root collection directory",
    )
    parser.add_argument(
        "--root-ref",
        type=rd.ArtifactRef.parse,
        required=True,
        help="exact root Data reference: data/slug@revision",
    )
    parser.add_argument(
        "--input-ref",
        type=rd.ArtifactRef.parse,
        required=True,
        help="exact same-root primary text Data reference",
    )
    parser.add_argument(
        "--output-ref",
        type=rd.ArtifactRef.parse,
        required=True,
        help="new aligned token Data reference; lengths are automatic",
    )
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        required=True,
        help="local Git-tracked tokenizer.json; never downloads",
    )
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="full Git commit containing the exact local asset bytes",
    )
    parser.add_argument(
        "--logging-dir",
        type=Path,
        required=True,
        help="native logs/completions, reused unchanged for compatible resume",
    )
    parser.add_argument(
        "--prefix-tokens",
        type=token_ids,
        required=True,
        help="comma-separated IDs inserted once before each body; empty string for none",
    )
    parser.add_argument(
        "--suffix-tokens",
        type=token_ids,
        required=True,
        help="comma-separated IDs inserted once after each body; empty string for none",
    )
    for flag, default, meaning in (
        ("tasks", 1, "total native partitions"),
        ("workers", 1, "concurrent processes; cannot exceed tasks"),
        (
            "threads",
            1,
            "Rayon threads per worker; budget workers*threads within allocation CPUs",
        ),
        ("batch-size", 1000, "maximum rows per encoding call"),
        (
            "batch-bytes",
            32 * 1024**2,
            "maximum input UTF-8 bytes per call; oversize rows run alone",
        ),
        (
            "max-sequences-per-shard",
            100000,
            "maximum sequences in one native BIN/IDX pair",
        ),
    ):
        parser.add_argument(
            f"--{flag}",
            type=positive_int,
            default=default,
            help=f"{meaning} (default: {default})",
        )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="permit existing output mutation and reuse compatible native tasks (default: fail on nonempty output)",
    )
    parser.add_argument(
        "--proposal-file",
        type=Path,
        help="complete ADT change-set JSON, declaring token Data and automatic lengths; omitted disables tracking",
    )
    parser.add_argument(
        "--proposal-repo",
        help="explicit GitHub owner/repository for tracked submission; requires --proposal-file and ADT SDK (Python 3.11+)",
    )
    parser.add_argument(
        "--proposal-issue",
        type=positive_int,
        help="open issue number to reuse for the exact whole record set; default discovers an unambiguous match",
    )
    try:
        values = vars(parser.parse_args(argv))
        file = values.pop("proposal_file")
        repo = values.pop("proposal_repo")
        issue = values.pop("proposal_issue")
        if bool(file) != bool(repo) or (issue is not None and file is None):
            raise ValueError(
                "--proposal-file and --proposal-repo are required together; --proposal-issue requires both"
            )
        if file is not None:
            # Preserve decimal values for ADT's lossless metadata contract.
            payload = json.loads(file.read_text(encoding="utf-8"), parse_float=Decimal)
            if not isinstance(payload, dict):
                raise ValueError("proposal JSON must be a complete change-set object")
            values["proposal"] = rd.TrackedProposal(
                payload=payload, repo=repo, issue=issue
            )
        return TokenizationJob(**values)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    """Execute the parsed local job and report validated row/token totals."""
    job = get_args(argv)
    try:
        tokens, _ = run_job(job)
    except ImportError as error:
        raise SystemExit(
            "tokenization dependency unavailable; tracked runs require Python 3.11+ "
            "and the coordinated ADT SDK (see the runbook)"
        ) from error
    except (OSError, ValueError) as error:
        raise SystemExit(f"tokenization failed: {error}") from error
    print(
        f"Ready: {tokens.member_ref}; {tokens.row_count} rows, {tokens.semantics['token_count']} stored tokens"
    )


if __name__ == "__main__":
    main()
