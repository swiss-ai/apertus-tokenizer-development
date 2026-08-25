"""Read the existing per-dataset shell configs without executing them."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .transforms import MINHASH_UPSAMPLING, TransformRequest, parse_transform_request

REQUIRED_KEYS = frozenset(
    {
        "TOKENIZER",
        "TOKENIZER_NAME",
        "DATASET_NAME",
        "COLUMN_KEY",
        "PATH_TO_RAW_DATASET",
        "PATH_TO_PREPROCESSING_METADATA",
        "PATH_TO_OUTPUT_FOLDER",
        "DUMPS_NUMBER",
    }
)


def _boolean(value: str, *, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"{key} must be true or false, got {value!r}")


def _assignments(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw_value = stripped.partition("=")
        if not separator or not key.replace("_", "").isalnum():
            raise ValueError(f"{path}:{line_number}: expected KEY=value")
        tokens = shlex.split(raw_value, comments=True, posix=True)
        values[key] = " ".join(tokens)
    return values


@dataclass(frozen=True)
class RunConfig:
    path: Path
    run: str
    dataset: str
    source: Path | None
    text_column: str
    extension: str
    tokenizer: str
    tokenizer_name: str
    metadata_root: Path
    output_root: Path
    dumps: int
    tasks: int
    transforms: tuple[TransformRequest, ...]

    @property
    def enabled(self) -> bool:
        return self.source is not None

    def plan(self) -> dict:
        return {
            "run": self.run,
            "dataset": self.dataset,
            "enabled": self.enabled,
            "config": str(self.path),
            "source": str(self.source) if self.source else None,
            "text_column": self.text_column,
            "output_root": str(self.output_root),
            "tokenizer": self.tokenizer,
            "dumps": self.dumps,
            "tasks_per_dump": self.tasks,
            "preprocessing": [request.descriptor() for request in self.transforms],
        }


def load_config(path: Path) -> RunConfig:
    values = _assignments(path)
    missing = sorted(REQUIRED_KEYS - values.keys())
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(missing)}")

    transform_values = values.get("PREPROCESSING_TRANSFORMS", "").split()
    rehydrate = _boolean(values.get("REHYDRATE_FLAG", "False"), key="REHYDRATE_FLAG")
    if rehydrate:
        if transform_values:
            raise ValueError(
                f"{path}: use PREPROCESSING_TRANSFORMS or REHYDRATE_FLAG, not both"
            )
        transform_values = [MINHASH_UPSAMPLING]
    source_value = values["PATH_TO_RAW_DATASET"].strip()
    dumps = int(values["DUMPS_NUMBER"])
    tasks = int(values.get("NUMBER_OF_DATATROVE_TASKS", "1"))
    if dumps < 1 or tasks < 1:
        raise ValueError(f"{path}: dump and task counts must be positive")
    return RunConfig(
        path=path.resolve(),
        run=path.stem,
        dataset=values["DATASET_NAME"],
        source=Path(source_value) if source_value else None,
        text_column=values["COLUMN_KEY"],
        extension=values.get("EXTENSION", ".parquet"),
        tokenizer=values["TOKENIZER"],
        tokenizer_name=values["TOKENIZER_NAME"],
        metadata_root=Path(values["PATH_TO_PREPROCESSING_METADATA"]),
        output_root=Path(values["PATH_TO_OUTPUT_FOLDER"]),
        dumps=dumps,
        tasks=tasks,
        transforms=tuple(parse_transform_request(item) for item in transform_values),
    )


def discover_configs(config_dir: Path) -> list[RunConfig]:
    configs = [load_config(path) for path in sorted(config_dir.glob("*.cfg"))]
    if not configs:
        raise ValueError(f"no .cfg files found in {config_dir}")
    return configs
