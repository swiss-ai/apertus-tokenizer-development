"""Submit root-aligned tokenization and publication as native Slurm jobs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from data_pipeline_pretrain.executor.completion import submit_with_finalizer
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from .tokenize import TokenizationJob, get_args, local_main, prepare_job


def launch_parser() -> argparse.ArgumentParser:
    """Describe scheduler settings shared by the ordinary and recording commands."""
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False
    )
    parser.add_argument(
        "--local", action="store_true", help="Run a single-machine smoke test"
    )
    parser.add_argument(
        "--partition", help="Slurm partition for tokenization and publication"
    )
    parser.add_argument("--account", help="Slurm allocation account")
    parser.add_argument(
        "--time",
        default="02:00:00",
        help="Slurm wall time per task (default: 02:00:00)",
    )
    parser.add_argument(
        "--env-command", help="Activate the shared Python environment and source paths"
    )
    parser.add_argument(
        "--mem-per-cpu-gb",
        type=int,
        default=2,
        help="Slurm memory per allocated CPU in GiB (default: 2)",
    )
    return parser


def executor_factory(
    job: TokenizationJob, settings: argparse.Namespace
) -> Callable[[list[PipelineStep], int, Path], SlurmPipelineExecutor]:
    """Build native task executors with one tokenizer process per Slurm task."""
    if not settings.partition or not settings.account or not settings.env_command:
        raise ValueError(
            "Slurm execution requires --partition, --account and --env-command"
        )
    if settings.mem_per_cpu_gb < 1:
        raise ValueError("--mem-per-cpu-gb must be positive")

    def native(
        pipeline: list[PipelineStep], tasks: int, logs: Path
    ) -> SlurmPipelineExecutor:
        return SlurmPipelineExecutor(
            pipeline=pipeline,
            tasks=tasks,
            workers=min(job.workers, tasks),
            time=settings.time,
            partition=settings.partition,
            qos="normal",
            cpus_per_task=job.threads,
            mem_per_cpu_gb=settings.mem_per_cpu_gb,
            sbatch_args={"account": settings.account, "nodes": 1, "ntasks": 1},
            env_command=(
                f"{settings.env_command}\n"
                f"export RAYON_NUM_THREADS={job.threads}\n"
                "export TOKENIZERS_PARALLELISM=true"
            ),
            logging_dir=str(logs),
            job_name="tokenize",
        )

    return native


def main(argv: list[str] | None = None) -> None:
    """Submit a worker array and dependent publication, or execute explicit local mode."""
    parser = launch_parser()
    settings, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        parser.print_help()
    if settings.local:
        local_main(remaining)
        return
    job = get_args(remaining)
    try:
        writer, native = prepare_job(job, executor_factory(job, settings))
        assert isinstance(native, SlurmPipelineExecutor)

        def finish() -> None:
            metadata = writer.finalize(native)
            print(f"Ready: {metadata.member_ref}; {metadata.row_count} rows")

        publication = submit_with_finalizer(native, finish)
        print(f"Submitted tokenization {native.job_id}; publication {publication}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"tokenization submission failed: {error}\n")


if __name__ == "__main__":
    main()
