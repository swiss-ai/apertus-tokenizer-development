"""Tokenize exact root-aligned Data with a pinned local tokenizer asset.

Run with ``python -m tokenization_scripts.tokenize --help``.
The job preserves root rows and writes packed tokens with exact stored lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pyarrow.compute as pc
from data_pipeline_pretrain.pipeline import root_data as rd
from data_pipeline_pretrain.pipeline.root_data.format import JSONObject, reference
from datatrove.data import Document
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from tokenizers import Tokenizer


@dataclass(frozen=True)
class TokenizationJob:
    """Exact input/output refs and per-task encoding controls.

    Tasks divide root rows into disjoint ranges; workers run tasks concurrently; threads
    bound each worker's Rayon pool. Encoding batches count rows and UTF-8 bytes.
    An oversize row runs alone without truncation.
    ``tokenizer_revision`` is the full Git commit containing ``tokenizer_file``.
    Task logs and all settings must match when the operator requests resume.
    ``data_store`` is the physical directory holding the collections; ``root_ref``
    identifies the alignment root: their shared row population and order.
    ``input_ref`` defaults to that root Data and may select another aligned Data.
    References accept ``slug@revision`` and are stored as exact Data references.
    Resume applies only to unfinished outputs.
    """

    data_store: Path
    root_ref: str
    output_ref: str
    tokenizer_file: Path
    tokenizer_revision: str
    logging_dir: Path
    prefix_tokens: tuple[int, ...]
    suffix_tokens: tuple[int, ...]
    input_ref: str | None = None
    text_column: str = "text"
    tasks: int = 1
    workers: int = 1
    threads: int = 1
    batch_size: int = 1000
    batch_bytes: int = 32 * 1024**2
    resume: bool = False

    def __post_init__(self) -> None:
        root_ref = reference(self.root_ref, kind="data")
        input_ref = (
            root_ref
            if self.input_ref is None
            else reference(self.input_ref, kind="data")
        )
        output_ref = reference(self.output_ref, kind="data")
        if output_ref in (root_ref, input_ref):
            raise ValueError("output must be a distinct aligned Data revision")
        object.__setattr__(self, "root_ref", root_ref)
        object.__setattr__(self, "input_ref", input_ref)
        object.__setattr__(self, "output_ref", output_ref)
        for name in (
            "tasks",
            "workers",
            "threads",
            "batch_size",
            "batch_bytes",
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


def load_asset(path: Path, revision: str) -> tuple[bytes, JSONObject]:
    """Verify local asset bytes against an exact Git commit, without network I/O.

    The asset must be tracked beneath a Git worktree with an origin URL. A local
    edit, untracked asset or wrong revision fails before output mutation. Git LFS
    files require materialized bytes matching the committed pointer's size/SHA-256.
    Returning verified bytes keeps worker loading independent of later file edits.
    """
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"local tokenizer asset is missing: {path}; restore revision {revision} "
            "and run git lfs pull in the asset checkout to materialize Git LFS files"
        ) from error
    repository = Path(
        _git(path.parent, "rev-parse", "--show-toplevel").decode().strip()
    )
    relative = path.relative_to(repository).as_posix()
    if _git(repository, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("tokenizer revision must identify a Git commit")
    pinned = _git(repository, "show", f"{revision}:{relative}")
    materialized = path.read_bytes()
    lfs_header = b"version https://git-lfs.github.com/spec/v1"
    if pinned.startswith(lfs_header):
        pointer = re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\r?\n"
            rb"oid sha256:([0-9a-f]{64})\r?\nsize (0|[1-9][0-9]*)\r?\n?",
            pinned,
        )
        if pointer is None:
            raise ValueError("committed tokenizer asset has an invalid Git LFS pointer")
        if materialized.startswith(lfs_header):
            command = shlex.join(
                ["git", "-C", str(repository), "lfs", "pull", f"--include={relative}"]
            )
            raise ValueError(
                "local tokenizer asset is an unmaterialized Git LFS pointer; "
                f"at revision {revision}, run: {command}"
            )
        expected_size = int(pointer[2])
        if len(materialized) != expected_size:
            raise ValueError(
                "local Git LFS tokenizer size differs from the configured Git revision: "
                f"expected {expected_size} bytes, found {len(materialized)}"
            )
        if hashlib.sha256(materialized).hexdigest() != pointer[1].decode("ascii"):
            raise ValueError(
                "local Git LFS tokenizer SHA-256 differs from the configured Git revision"
            )
    elif materialized != pinned:
        raise ValueError(
            "local tokenizer bytes differ from the configured Git revision"
        )
    # get-url expands machine insteadOf rewrites, which may contain credentials.
    origin = _repository_url(
        _git(repository, "config", "--get", "remote.origin.url").decode().strip()
    )
    return materialized, {"repository": origin, "commit": revision, "path": relative}


class LocalBodyEncoder:
    """Load verified local JSON and encode bodies with special tokens disabled.

    Asset truncation and padding are explicitly disabled. One output body
    corresponds to every input text; the next encoding step frames it once.
    """

    def __init__(self, asset: bytes) -> None:
        """Retain verified bytes; each worker initializes its native model lazily."""
        self.asset = asset
        self._tokenizer: Tokenizer | None = None

    def __repr__(self) -> str:
        """Keep the verified asset digest in executor logs without copying its JSON."""
        return f"LocalBodyEncoder(sha256={hashlib.sha256(self.asset).hexdigest()})"

    def __getstate__(self) -> dict[str, bytes]:
        """Transfer verified bytes; cached native tokenizers stay in their process."""
        return {"asset": self.asset}

    def __setstate__(self, state: dict[str, bytes]) -> None:
        """Restore the encoder with lazy native-model initialization."""
        self.asset = state["asset"]
        self._tokenizer = None

    @property
    def tokenizer(self) -> Tokenizer:
        """Reuse this instance's model without transporting a descriptor lock."""
        if self._tokenizer is None:
            tokenizer = Tokenizer.from_str(self.asset.decode("utf-8"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            self._tokenizer = tokenizer
        return self._tokenizer

    def __call__(self, texts: Sequence[str]) -> list[list[int]]:
        """Return one list of body token IDs per input text without tracking offsets."""
        encode = getattr(
            self.tokenizer, "encode_batch_fast", self.tokenizer.encode_batch
        )
        return [row.ids for row in encode(list(texts), add_special_tokens=False)]


class EncodeTokens(PipelineStep):
    """Encode text batches into complete Parquet token cells and stored lengths.

    ``batch_size`` counts rows and ``batch_bytes`` counts input UTF-8 bytes. A
    larger row runs alone without truncation. Prefix/suffix IDs in ``spec`` are
    inserted once. Every Document retains its text, ID and root coordinates.
    """

    name = "Encode tokens"
    type = "Tokenizer"

    def __init__(
        self,
        asset: bytes,
        spec: rd.TokenSpec,
        *,
        batch_size: int = 1000,
        batch_bytes: int = 32 * 1024**2,
    ) -> None:
        super().__init__()
        if batch_size < 1 or batch_bytes < 1:
            raise ValueError("encoding batch limits must be positive")
        self.encoder = LocalBodyEncoder(asset)
        self.spec = spec
        self.batch_size, self.batch_bytes = batch_size, batch_bytes

    def _batches(self, data: Iterable[Document]) -> Iterator[list[Document]]:
        batch: list[Document] = []
        size = 0
        for document in data:
            if not isinstance(document.text, str):
                raise TypeError("token input text must be a string")
            row_bytes = len(document.text.encode("utf-8"))
            if batch and (
                len(batch) >= self.batch_size or size + row_bytes > self.batch_bytes
            ):
                yield batch
                batch, size = [], 0
            batch.append(document)
            size += row_bytes
        if batch:
            yield batch

    def run(
        self, data: Iterable[Document], rank: int = 0, world_size: int = 1
    ) -> Iterator[Document]:
        """Validate each encoded batch, then attach tokens and intrinsic lengths."""
        for batch in self._batches(data):
            bodies = self.encoder([document.text for document in batch])
            if len(bodies) != len(batch):
                raise ValueError("encoder must return exactly one body per input")
            encoded = [rd.encode_tokens(body, self.spec) for body in bodies]
            for document, fields in zip(batch, encoded):
                document.metadata.update(fields)
                yield document


def prepare_job(
    job: TokenizationJob,
    executor_factory: (
        Callable[[list[PipelineStep], int, Path], PipelineExecutor] | None
    ) = None,
) -> tuple[rd.RootWriter, PipelineExecutor]:
    """Verify assets and reserve aligned token Data with its native executor.

    Set the process tokenizer thread environment and reserve pending collection
    metadata with the executor's task geometry and logging directory. Return the
    prepared writer and executor. Finalize after native completion: Slurm needs
    a dependent publication job; local execution is synchronous. Asset, input and
    occupied-output errors fail before execution. Explicit resume reuses
    compatible unfinished tasks.
    """
    asset, identity = load_asset(job.tokenizer_file, job.tokenizer_revision)
    os.environ["RAYON_NUM_THREADS"] = str(job.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    vocabulary_size = Tokenizer.from_str(asset.decode("utf-8")).get_vocab_size(
        with_added_tokens=True
    )
    spec = rd.TokenSpec(
        encoding={
            "tokenizer": identity,
            "add_special_tokens": False,
            "truncation": None,
            "padding": None,
        },
        vocabulary_size=vocabulary_size,
        prefix_tokens=job.prefix_tokens,
        suffix_tokens=job.suffix_tokens,
    )
    source = rd.RootParquetReader(
        job.data_store,
        job.root_ref,
        {job.input_ref: {"text": job.text_column}},
    )
    writer = rd.RootWriter(
        job.data_store,
        job.root_ref,
        data=job.output_ref,
        source=source,
        schema=rd.token_schema(spec),
        batch_size=job.batch_size,
    )
    pipeline: list[PipelineStep] = [
        source,
        EncodeTokens(
            asset, spec, batch_size=job.batch_size, batch_bytes=job.batch_bytes
        ),
        writer,
    ]
    executor = (
        executor_factory(pipeline, job.tasks, job.logging_dir)
        if executor_factory
        else LocalPipelineExecutor(
            pipeline,
            tasks=job.tasks,
            workers=job.workers,
            logging_dir=str(job.logging_dir),
            start_method="spawn",
        )
    )
    writer.prepare(executor, resume=job.resume)
    return writer, executor


def run_job(job: TokenizationJob) -> rd.CollectionMetadata:
    """Write aligned token Data using verified assets and native task completion.

    Input, encoding, alignment and corruption errors fail the job. Explicit resume
    reuses compatible unfinished tasks. Completed outputs are immutable.
    """
    writer, executor = prepare_job(job)
    executor.run()
    return writer.finalize(executor)


def positive_int(value: str) -> int:
    """Argparse positive integer, used for counts and byte limits."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def data_reference(value: str) -> str:
    """Canonicalize a Data identity and report invalid CLI input clearly."""
    try:
        return reference(value, kind="data")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        "--data-store",
        type=Path,
        required=True,
        help="directory holding durable Data and Annotation collections",
    )
    parser.add_argument(
        "--root-ref",
        type=data_reference,
        required=True,
        help="alignment-root Data ID: slug@revision",
    )
    parser.add_argument(
        "--input-ref",
        type=data_reference,
        help="aligned text Data ID: slug@revision (default: --root-ref)",
    )
    parser.add_argument(
        "--output-ref",
        type=data_reference,
        required=True,
        help="new token Data ID: slug@revision",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="input string column to encode (default: text)",
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
        help="full Git commit containing the exact local tokenizer asset bytes",
    )
    parser.add_argument(
        "--logging-dir",
        type=Path,
        required=True,
        help="task logs and completion records, reused for compatible resume",
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
        ("tasks", 1, "total task ranges over the root rows"),
        ("workers", 1, "concurrent tasks or local processes; cannot exceed tasks"),
        (
            "threads",
            1,
            "Rayon threads per worker; budget workers*threads within allocation CPUs",
        ),
        ("batch-size", 1000, "maximum rows per encoding call and physical write batch"),
        (
            "batch-bytes",
            32 * 1024**2,
            "maximum input UTF-8 bytes per encoding call; oversize rows run alone",
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
        help="resume compatible unfinished output and reuse Datatrove tasks; completed output is immutable",
    )
    try:
        values = vars(parser.parse_args(argv))
        return TokenizationJob(**values)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def local_main(argv: list[str] | None = None) -> None:
    """Run an explicit single-machine job and report validated row/token totals."""
    job = get_args(argv)
    try:
        tokens = run_job(job)
        collection = rd.ParquetCollection(
            job.data_store, job.root_ref, job.output_ref, batch_size=job.batch_size
        )
        token_count = sum(
            pc.sum(batch.column("stored_tokens")).as_py() or 0
            for batch in collection.batches(["stored_tokens"])
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"tokenization failed: {error}") from error
    print(
        f"Ready: {tokens.member_ref}; {tokens.row_count} rows, "
        f"{token_count} stored tokens"
    )


def main(argv: list[str] | None = None) -> None:
    """Submit native Slurm tasks, or run an explicitly requested local smoke test."""
    from .slurm import main as launch

    launch(argv)


if __name__ == "__main__":
    main()
