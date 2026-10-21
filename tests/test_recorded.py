"""ADT boundary checks use its real prepared-run object and the ordinary job."""

import json
from dataclasses import replace
from types import SimpleNamespace

import dill
import pytest
from data_pipeline_pretrain.pipeline import root_data as rd
from test_tokenization_scripts import prepare, sequences

from tokenization_scripts import recorded

try:
    from adt import recording
    from adt.recording import PreparedRun
except ModuleNotFoundError as error:
    if error.name != "adt":
        raise
    pytest.skip("optional ADT SDK is not installed", allow_module_level=True)


def test_recording_binds_exact_producer_and_submits_after_success(
    tmp_path, monkeypatch
):
    events = []
    expected = {"repository": "https://example.org/tokenizer.git", "commit": "a" * 40}
    metadata = object()
    logs = tmp_path / "logs"

    def prepare(refs, *, repo, code_directory):
        assert refs == ["data/tokens@v1"] and repo == "person/catalogue"
        assert code_directory == recorded.Path(recorded.__file__).parent
        events.append("capture")
        return PreparedRun(
            expected["repository"],
            expected["commit"],
            lambda: events.append("submit") or "https://example.org/issue",
        )

    def run():
        assert json.loads((logs / "producer.json").read_text()) == expected
        events.append("run")

    executor = SimpleNamespace(logging_dir=SimpleNamespace(path=logs), run=run)

    def finalize(native):
        assert native is executor
        events.append("finalize")
        return metadata

    def prepare_job(job):
        events.append("prepare")
        assert not logs.exists()
        return SimpleNamespace(finalize=finalize), executor

    monkeypatch.setattr(recording, "prepare_run", prepare)
    monkeypatch.setattr(recorded, "prepare_job", prepare_job)
    job = SimpleNamespace(output_ref="data/tokens@v1")
    assert recorded.run_recorded(job, adt_repo="person/catalogue") == (
        metadata,
        "https://example.org/issue",
    )
    assert events == ["capture", "prepare", "run", "finalize", "submit"]


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("producer differs"),
        FileExistsError("completed"),
        OSError("failed write"),
    ],
)
def test_failed_job_never_submits_an_issue(monkeypatch, failure):
    submitted = []
    monkeypatch.setattr(
        recording,
        "prepare_run",
        lambda *args, **kwargs: PreparedRun(
            "https://example.org/t.git", "a" * 40, lambda: submitted.append(True)
        ),
    )

    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(recorded, "prepare_job", run)
    job = SimpleNamespace(output_ref="data/tokens@v1")
    with pytest.raises(type(failure)):
        recorded.run_recorded(job)
    assert submitted == []


def test_cli_preserves_all_ordinary_job_arguments(monkeypatch, capsys):
    arguments = ["--tasks", "3", "--text-column", "body", "--resume"]
    job = object()
    metadata = SimpleNamespace(member_ref="data/tokens@v1", row_count=4)

    def parse(actual):
        assert actual == arguments
        return job

    def run(actual, *, adt_repo):
        assert actual is job and adt_repo == "person/catalogue"
        return metadata, "https://example.org/issue"

    monkeypatch.setattr(recorded, "get_args", parse)
    monkeypatch.setattr(recorded, "run_recorded", run)
    recorded.main(["--local", "--adt-repo", "person/catalogue", *arguments])
    assert "4 rows" in capsys.readouterr().out


def test_recorded_job_keeps_producer_and_completed_data_on_issue_failure(
    tmp_path, monkeypatch
):
    job = prepare(tmp_path, ["", "a b"])
    job = replace(
        job,
        root_ref=job.root_ref.split("/", 1)[1],
        output_ref=job.output_ref.split("/", 1)[1],
    )
    producer = {"repository": "https://example.org/tokenizer.git", "commit": "a" * 40}
    unavailable = True

    def submit():
        assert (
            rd.read_metadata(job.data_store, job.root_ref, job.output_ref).result
            is not None
        )
        assert json.loads((job.logging_dir / "producer.json").read_text()) == producer
        if unavailable:
            raise OSError("issue service unavailable")
        return "https://example.org/issue"

    def capture(refs, **kwargs):
        assert refs == ["data/offline-tokens@v1"]
        return PreparedRun(producer["repository"], producer["commit"], submit)

    monkeypatch.setattr(recording, "prepare_run", capture)
    with pytest.raises(
        RuntimeError, match="is complete; ADT recording failed.*issue service"
    ):
        recorded.run_recorded(job)
    with pytest.raises(ValueError, match="native completions require.*explicit resume"):
        recorded.run_recorded(job)
    assert [row.tolist() for row in sequences(job)] == [[1, 2], [1, 3, 4, 2]]
    output = rd.collection_path(job.data_store, job.root_ref, job.output_ref)
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    producer_record = (job.logging_dir / "producer.json").read_bytes()
    unavailable = False
    assert (
        recording.prepare_run([job.output_ref]).submit() == "https://example.org/issue"
    )
    assert all(p.read_bytes() == content for p, content in before.items())
    assert (job.logging_dir / "producer.json").read_bytes() == producer_record


def test_recorded_help_includes_the_ordinary_job_arguments(capsys):
    with pytest.raises(SystemExit) as error:
        recorded.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--adt-repo" in help_text
    assert "--text-column" in help_text
    assert "--output-ref" in help_text


def test_pending_recorded_job_requires_original_producer_before_native_retry(
    tmp_path, monkeypatch
):
    job = prepare(tmp_path, ["a", "b"])
    captured = {"repository": "https://example.org/tokenizer.git", "commit": "a" * 40}
    submitted = []
    monkeypatch.setattr(
        recording,
        "prepare_run",
        lambda *args, **kwargs: PreparedRun(
            captured["repository"],
            captured["commit"],
            lambda: submitted.append(True) or "https://example.org/issue",
        ),
    )
    finalize = rd.RootWriter.finalize

    def interrupt(writer, native):
        raise OSError("finalization interrupted")

    monkeypatch.setattr(rd.RootWriter, "finalize", interrupt)
    with pytest.raises(OSError, match="finalization interrupted"):
        recorded.run_recorded(job)
    output = rd.collection_path(job.data_store, job.root_ref, job.output_ref)
    before = {p: p.read_bytes() for p in output.iterdir() if p.is_file()}
    record = job.logging_dir / "producer.json"
    original = record.read_bytes()
    captured["commit"] = "b" * 40
    monkeypatch.setattr(rd.RootWriter, "finalize", finalize)
    with pytest.raises(ValueError, match="original producer"):
        recorded.run_recorded(replace(job, resume=True))
    assert record.read_bytes() == original
    assert all(p.read_bytes() == value for p, value in before.items())
    assert submitted == []
    captured["commit"] = "a" * 40
    record.unlink()
    with pytest.raises(ValueError, match="no recorded producer"):
        recorded.run_recorded(replace(job, resume=True))
    assert not record.exists()
    record.write_bytes(original)
    metadata, url = recorded.run_recorded(replace(job, resume=True))
    assert metadata.row_count == 2 and url == "https://example.org/issue"
    assert record.read_bytes() == original
    assert all(
        p.read_bytes() == value for p, value in before.items() if p.suffix == ".parquet"
    )
    assert submitted == [True]


def test_recorded_asset_preflight_failure_leaves_no_producer_binding(
    tmp_path, monkeypatch
):
    job = prepare(tmp_path, ["a"])
    monkeypatch.setattr(
        recording,
        "prepare_run",
        lambda *args, **kwargs: PreparedRun(
            "https://example.org/tokenizer.git", "a" * 40, lambda: "unused"
        ),
    )
    job.tokenizer_file.write_text("invalid local tokenizer edit")
    with pytest.raises(ValueError, match="bytes differ"):
        recorded.run_recorded(job)
    assert not (job.logging_dir / "producer.json").exists()
    assert not rd.collection_path(job.data_store, job.root_ref, job.output_ref).exists()


def test_slurm_records_after_native_token_data_completion(tmp_path, monkeypatch):
    import argparse

    from datatrove.executor.slurm import SlurmPipelineExecutor

    from tokenization_scripts.slurm import executor_factory

    job = prepare(tmp_path, ["", "a b"])
    issue = tmp_path / "issue"
    available = tmp_path / "recording-available"
    scheduled = []

    def submit():
        if not available.exists():
            raise OSError("recording unavailable")
        issue.write_text("recorded")
        return "https://example.org/issue"

    monkeypatch.setattr(
        recording, "prepare_run",
        lambda *args, **kwargs: PreparedRun(
            "https://example.org/tokenizer.git", "a" * 40,
            submit,
        ),
    )

    def launch(native):
        scheduled.append(native)
        native.job_id = "123"

    monkeypatch.setattr(SlurmPipelineExecutor, "launch_job", launch)
    factory = executor_factory(job, argparse.Namespace(
        partition="debug", account="test", time="00:03:00",
        env_command=". /shared/venv/bin/activate", mem_per_cpu_gb=2,
    ))
    assert recorded.run_recorded(job, executor_factory=factory) == "123"
    assert not issue.exists()
    (publication,) = scheduled
    payload = dill.dumps(publication.pipeline[0])
    work = publication.depends
    assert rd.read_metadata(
        job.data_store, job.root_ref, job.output_ref, require_complete=False,
    ).result is None
    with pytest.raises(ValueError, match="completion"):
        list(dill.loads(payload)(None, 0, 1))
    for rank in range(work.tasks):
        work._run_for_rank(rank)
    with pytest.raises(RuntimeError, match="is complete; ADT recording failed"):
        list(dill.loads(payload)(None, 0, 1))
    output = rd.collection_path(job.data_store, job.root_ref, job.output_ref)
    before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    available.touch()
    assert list(dill.loads(payload)(None, 0, 1)) == []
    assert issue.read_text() == "recorded"
    assert all(path.read_bytes() == content for path, content in before.items())
    assert [row.tolist() for row in sequences(job)] == [[1, 2], [1, 3, 4, 2]]
