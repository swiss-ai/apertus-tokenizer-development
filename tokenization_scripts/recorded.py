"""Record a successful tokenization job with its exact producer commit in ADT.

Pass the tokenization job arguments, plus optional --adt-repo.
Install tests/requirements/adt.txt using Python >=3.11 before using this command.
Use python -m tokenization_scripts.tokenize --help for job arguments.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from data_pipeline_pretrain.executor.completion import submit_with_finalizer
from data_pipeline_pretrain.pipeline import root_data as rd
from data_pipeline_pretrain.transparency import bind_producer
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from .tokenize import TokenizationJob, get_args, prepare_job


def run_recorded(
    job: TokenizationJob,
    *,
    adt_repo: str | None = None,
    executor_factory: (
        Callable[[list[PipelineStep], int, Path], PipelineExecutor] | None
    ) = None,
) -> tuple[rd.CollectionMetadata, str] | str:
    """Capture the producer, execute the ordinary job and submit its run record.

    ADT refreshes accepted metadata and reuses matching pending issues. Input,
    encoding, output-occupation and producer errors prevent issue submission.
    Explicit pending resume requires the original producer checkout. If submission
    fails after physical completion, record through ADT from that same checkout;
    the completed token Data remains immutable. A Slurm factory schedules native
    work and returns its dependent publication job ID; default execution is local.
    """
    from adt.recording import prepare_run

    recording = prepare_run(
        [job.output_ref], repo=adt_repo, code_directory=Path(__file__).parent
    )
    writer, executor = (
        prepare_job(job, executor_factory) if executor_factory else prepare_job(job)
    )
    bind_producer(
        executor.logging_dir.path,
        {"repository": recording.repository, "commit": recording.commit},
    )

    def finish() -> tuple[rd.CollectionMetadata, str]:
        metadata = writer.finalize(executor)
        try:
            return metadata, recording.submit()
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Token Data {job.output_ref} is complete; ADT recording failed: {error}. "
                "From the same producer checkout, retry prepare_run(...).submit() "
                "to submit its record."
            ) from error

    if isinstance(executor, SlurmPipelineExecutor):

        def publish() -> None:
            metadata, url = finish()
            print(f"Completed {metadata.member_ref}: {metadata.row_count} rows\n{url}")

        return submit_with_finalizer(executor, publish)
    executor.run()
    return finish()


def main(argv: list[str] | None = None) -> None:
    """Accept ordinary job flags and the optional issue destination override."""
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False
    )
    parser.add_argument(
        "--adt-repo",
        help="recording destination owner/repository; default: shared ADT repository",
    )
    from .slurm import executor_factory, launch_parser

    recording, remaining = parser.parse_known_args(argv)
    scheduling = launch_parser()
    settings, remaining = scheduling.parse_known_args(remaining)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
        scheduling.print_help()
    job = get_args(remaining)
    try:
        result = (
            run_recorded(job, adt_repo=recording.adt_repo)
            if settings.local
            else run_recorded(
                job,
                adt_repo=recording.adt_repo,
                executor_factory=executor_factory(job, settings),
            )
        )
        if isinstance(result, str):
            print(f"Submitted publication {result}")
        else:
            metadata, url = result
            print(f"Completed {metadata.member_ref}: {metadata.row_count} rows\n{url}")
    except ImportError as error:
        parser.exit(
            1,
            f"ADT recording needs Python >=3.11 and tests/requirements/adt.txt: {error}\n",
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
