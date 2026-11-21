"""Offline policy compilation. Inference accepts detector events alone."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Literal

import numpy as np
import pymatching
import stim
from jsonschema import Draft202012Validator

DISTANCES = (3, 5, 7)
SCALES = (0.75, 1.0, 1.25)
Kind = Literal["mwpm", "correlated_mwpm", "weighted_mwpm"]


@cache
def validator(name: str) -> Draft202012Validator:
    schema = json.loads(files("bascd").joinpath("schemas", name + ".json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True)
class DecoderSpec:
    decoder_type: Kind
    temporal_scale: float = 1.0
    boundary_scale: float = 1.0

    def __post_init__(self):
        if self.decoder_type not in ("mwpm", "correlated_mwpm", "weighted_mwpm"):
            raise ValueError("unknown decoder type")
        for scale in (self.temporal_scale, self.boundary_scale):
            if type(scale) not in (int, float) or scale not in SCALES:
                raise ValueError(f"weight scales must be chosen from {SCALES}")
        adjusted = (self.temporal_scale, self.boundary_scale) != (1, 1)
        if adjusted != (self.decoder_type == "weighted_mwpm"):
            raise ValueError("only weighted_mwpm accepts adjustments and requires at least one")

    def to_dict(self):
        result = {"decoder_type": self.decoder_type}
        if self.decoder_type == "weighted_mwpm":
            result.update(temporal_scale=float(self.temporal_scale), boundary_scale=float(self.boundary_scale))
        return result


CHOICES = (DecoderSpec("mwpm"), DecoderSpec("correlated_mwpm")) + tuple(
    DecoderSpec("weighted_mwpm", temporal, boundary)
    for temporal, boundary in itertools.product(SCALES, repeat=2) if (temporal, boundary) != (1, 1)
)


@dataclass(frozen=True)
class Policy:
    specs: tuple[DecoderSpec, DecoderSpec, DecoderSpec]

    def __post_init__(self):
        if type(self.specs) is not tuple or len(self.specs) != 3 or not all(isinstance(s, DecoderSpec) for s in self.specs):
            raise ValueError("a policy needs one decoder specification for each distance")

    @classmethod
    def from_dict(cls, value):
        validator("proposal").validate(value)
        return cls(tuple(DecoderSpec(**value["by_distance"][str(d)]) for d in DISTANCES))

    def at(self, distance):
        return self.specs[DISTANCES.index(distance)]

    def to_dict(self):
        return {"by_distance": {str(d): self.at(d).to_dict() for d in DISTANCES}}

    @property
    def policy_id(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass
class CompiledDecoder:
    graph: pymatching.Matching
    correlated: bool

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        return self.graph.decode_batch(detectors, enable_correlations=self.correlated)

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        return self.graph.decode(detectors, enable_correlations=self.correlated)


def compile_decoder(model: stim.DetectorErrorModel, spec: DecoderSpec) -> CompiledDecoder:
    correlated = spec.decoder_type == "correlated_mwpm"
    graph = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
    if spec.decoder_type == "weighted_mwpm":
        coords = model.get_detector_coordinates()
        changed = set()
        for a, b, edge in graph.edges():
            if b is None:
                group, scale = "boundary", spec.boundary_scale
            elif len(coords.get(a, ())) < 3 or len(coords.get(b, ())) < 3:
                raise ValueError("weighted policy requires detector coordinates (x, y, round)")
            elif coords[a][:2] == coords[b][:2] and coords[a][2] != coords[b][2]:
                group, scale = "temporal", spec.temporal_scale
            else:
                continue
            if scale == 1 or edge["weight"] == 0:
                continue
            changed.add(group)
            params = dict(weight=edge["weight"] * scale, fault_ids=edge["fault_ids"],
                          error_probability=edge["error_probability"], merge_strategy="replace")
            if b is None:
                graph.add_boundary_edge(a, **params)
            else:
                graph.add_edge(a, b, **params)
        for group, scale in (("boundary", spec.boundary_scale), ("temporal", spec.temporal_scale)):
            if scale != 1 and group not in changed:
                raise ValueError(f"no adjustable {group} edges in this model")
    return CompiledDecoder(graph, correlated)
