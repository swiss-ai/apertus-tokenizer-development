"""Explicit registry for transforms that are allowed inside tokenization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MINHASH_UPSAMPLING = "sampling.minhash_cluster_upsampling"
KNOWN_TRANSFORMS = frozenset({MINHASH_UPSAMPLING})


@dataclass(frozen=True)
class TransformRequest:
    type: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in KNOWN_TRANSFORMS:
            raise ValueError(f"unknown tokenization transform: {self.type!r}")
        if not isinstance(self.parameters, dict):
            raise TypeError("transform parameters must be an object")

    def descriptor(self) -> dict[str, Any]:
        return {"type": self.type, "version": 1, "parameters": self.parameters}


def parse_transform_request(value: str) -> TransformRequest:
    """Parse ``type`` or ``type=<compact JSON object>`` from a config/CLI."""
    transform_type, separator, raw_parameters = value.partition("=")
    parameters = json.loads(raw_parameters) if separator else {}
    return TransformRequest(transform_type, parameters)


def build_transforms(requests: list[TransformRequest]):
    """Instantiate the bounded registry without importing dataset renderers."""
    from data_pipeline_pretrain.pipeline.transforms import MinhashClusterUpsampler

    transforms = []
    for request in requests:
        if request.type == MINHASH_UPSAMPLING:
            parameters = dict(request.parameters)
            if "weights" in parameters:
                parameters["weights"] = {
                    int(limit): copies
                    for limit, copies in parameters["weights"].items()
                }
            transforms.append(MinhashClusterUpsampler(**parameters))
    return transforms
