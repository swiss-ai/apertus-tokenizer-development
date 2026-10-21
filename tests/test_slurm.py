"""Tokenization uses native Slurm tasks and publishes after completion."""

import dill
import pytest
from data_pipeline_pretrain.pipeline import root_data as rd
from datatrove.executor.slurm import SlurmPipelineExecutor
from test_tokenization_scripts import prepare, sequences

from tokenization_scripts.slurm import executor_factory, launch_parser
from tokenization_scripts.tokenize import prepare_job


@pytest.mark.parametrize("memory_per_cpu", [None, 6])
def test_native_factory_preserves_tokenization_and_allocation(
    tmp_path, memory_per_cpu
):
    job = prepare(tmp_path, ["", "a b"])
    arguments = [
        "--partition",
        "debug",
        "--account",
        "test",
        "--time",
        "00:03:00",
        "--env-command",
        ". /shared/venv/bin/activate",
    ]
    if memory_per_cpu is not None:
        arguments += ["--mem-per-cpu-gb", str(memory_per_cpu)]
    settings = launch_parser().parse_args(arguments)
    writer, work = prepare_job(job, executor_factory(job, settings))
    assert isinstance(work, SlurmPipelineExecutor)
    assert work.tasks == job.tasks and work.workers == job.workers
    assert work.cpus_per_task == job.threads
    assert work.get_sbatch_args()["mem-per-cpu"] == f"{memory_per_cpu or 2}G"
    assert f"export RAYON_NUM_THREADS={job.threads}" in work.env_command
    assert "export TOKENIZERS_PARALLELISM=true" in work.env_command
    assert (
        rd.read_metadata(
            job.data_store, job.root_ref, job.output_ref, require_complete=False
        ).result
        is None
    )
    for rank in range(job.tasks):
        work._run_for_rank(rank)
    metadata = writer.finalize(work)
    assert metadata.row_count == 2
    assert [row.tolist() for row in sequences(job)] == [[1, 2], [1, 3, 4, 2]]


def test_recovered_publisher_preserves_completed_token_data(tmp_path, monkeypatch):
    from tokenization_scripts import slurm

    job = prepare(tmp_path, ["", "a b"])
    scheduled = []

    def launch(native):
        scheduled.append(native)
        native.job_id = "123"

    monkeypatch.setattr(SlurmPipelineExecutor, "launch_job", launch)
    monkeypatch.setattr(slurm, "get_args", lambda arguments: job)
    slurm.main(
        [
            "--partition",
            "debug",
            "--account",
            "test",
            "--env-command",
            ". /shared/venv/bin/activate",
        ]
    )
    (publication,) = scheduled
    payload = dill.dumps(publication.pipeline[0])
    with pytest.raises(ValueError, match="completion"):
        list(dill.loads(payload)(None, 0, 1))
    work = publication.depends
    for rank in range(work.tasks):
        work._run_for_rank(rank)
    assert list(dill.loads(payload)(None, 0, 1)) == []
    output = rd.collection_path(job.data_store, job.root_ref, job.output_ref)
    before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert list(dill.loads(payload)(None, 0, 1)) == []
    assert all(path.read_bytes() == content for path, content in before.items())
    assert [row.tolist() for row in sequences(job)] == [[1, 2], [1, 3, 4, 2]]
