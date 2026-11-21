"""Vote across matching hypotheses that use the same detector record."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import stim
from scipy.special import expit

from .bath_prior import BathPrior, bath_prior_dem
from .decoders import CHOICES, CompiledDecoder, compile_decoder

ModelSource = Literal["native", "clock", "selected"]


@dataclass(frozen=True)
class EnsembleMember:
    source: ModelSource = "native"
    jitter: float = 0.0
    seed: int = 0
    correlated: bool = True

    def __post_init__(self):
        if self.source not in ("native", "clock", "selected"):
            raise ValueError("unknown matching model source")
        if type(self.jitter) not in (float, int) or not math.isfinite(self.jitter) or self.jitter < 0:
            raise ValueError("jitter must be a finite nonnegative log-odds standard deviation")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64 or (self.jitter == 0 and self.seed):
            raise ValueError("expected an unsigned 64-bit seed, zero when jitter is disabled")
        if type(self.correlated) is not bool:
            raise ValueError("correlated must be a boolean")


@dataclass(frozen=True)
class EnsembleSpec:
    members: tuple[EnsembleMember, ...]

    def __post_init__(self):
        if (type(self.members) is not tuple or not len(self.members) % 2 or
                not all(isinstance(member, EnsembleMember) for member in self.members)):
            raise ValueError("a majority vote needs an odd number of matching members")
        if len(set(self.members)) != len(self.members):
            raise ValueError("ensemble members must be distinct")

    def to_dict(self) -> dict:
        return {"members": [asdict(member) for member in self.members]}

    @classmethod
    def from_dict(cls, value: Mapping) -> "EnsembleSpec":
        return cls(tuple(EnsembleMember(**member) for member in value["members"]))

    def vote(self, predictions: Mapping[EnsembleMember, np.ndarray]) -> np.ndarray:
        """Score this specification using cached predictions from its members."""
        votes = np.sum([predictions[member] for member in self.members], axis=0)
        return (votes > len(self.members) // 2).astype(np.uint8)


FAST_ENSEMBLE = EnsembleSpec((
    EnsembleMember(correlated=False),
    EnsembleMember(),
    EnsembleMember("selected"),
))
BATH_ENSEMBLE = EnsembleSpec((
    EnsembleMember(),
    EnsembleMember("clock"),
    EnsembleMember("selected"),
))
FIVE_MEMBER_ENSEMBLE = EnsembleSpec((
    *BATH_ENSEMBLE.members,
    EnsembleMember(jitter=1.1, seed=11),
    EnsembleMember(jitter=1.1, seed=29),
))


def jitter_model(
    model: stim.DetectorErrorModel, standard_deviation: float, seed: int,
) -> stim.DetectorErrorModel:
    """Perturb each complete error instruction in log-odds space.

    Keep correlated components together and give faults in distinct rounds
    independent offsets by applying the perturbation after expanding repeats.
    """
    flat = model.flattened()
    offsets = np.random.default_rng(seed).normal(0, standard_deviation, flat.num_errors)
    result = stim.DetectorErrorModel()
    error_index = 0
    for instruction in flat:
        if instruction.type != "error":
            result.append(instruction)
            continue
        probability, = instruction.args_copy()
        offset = offsets[error_index]
        error_index += 1
        if 0 < probability < 0.5 and offset != 0:
            # Correlated PyMatching requires individual probabilities <= 1/2.
            probability = min(0.5, expit(math.log(probability) - math.log1p(-probability) + offset))
        result.append("error", probability, instruction.targets_copy(), tag=instruction.tag)
    return result


class PreparedEnsembleCase:
    """Reuse hypothesis DEMs while allowing independent matcher setup per run."""

    def __init__(
        self, circuit: stim.Circuit, native_model: stim.DetectorErrorModel, *,
        basis: str, distance: int, rounds: int, p: float, priors: Mapping[str, BathPrior],
    ):
        if (native_model.num_detectors, native_model.num_observables) != (circuit.num_detectors, 1):
            raise ValueError("expected a single-observable model matching the detector record")
        if circuit.num_observables != 1:
            raise ValueError("ensemble voting expects a single observable")
        self.circuit = circuit
        self.native_model = native_model
        self.parameters = dict(basis=basis, distance=distance, rounds=rounds, p=p)
        self.priors = dict(priors)
        self._base_models = {"native": native_model}
        self._perturbed_models: dict[tuple[str, float, int], stim.DetectorErrorModel] = {}

    def model_for(self, member: EnsembleMember) -> stim.DetectorErrorModel:
        if member.source not in self._base_models:
            if member.source not in self.priors:
                raise ValueError(f"missing bath prior for {member.source!r}")
            prior = self.priors[member.source]
            if prior.channel != member.source:
                raise ValueError(f"bath prior channel does not match {member.source!r}")
            model = bath_prior_dem(self.circuit, **self.parameters, prior=prior)
            if (model.num_detectors != self.native_model.num_detectors or model.num_observables != 1 or
                    model.get_detector_coordinates() != self.native_model.get_detector_coordinates()):
                raise ValueError("bath model does not match native detector ordering")
            self._base_models[member.source] = model
        model = self._base_models[member.source]
        if not member.jitter:
            return model
        key = (member.source, member.jitter, member.seed)
        if key not in self._perturbed_models:
            self._perturbed_models[key] = jitter_model(model, member.jitter, member.seed)
        return self._perturbed_models[key]

    def compile_member(self, member: EnsembleMember) -> CompiledDecoder:
        return compile_decoder(self.model_for(member), CHOICES[int(member.correlated)])

    def compile(self, spec: EnsembleSpec) -> "EnsembleDecoder":
        members = tuple(self.compile_member(member) for member in spec.members)
        return EnsembleDecoder(members, self.native_model.num_detectors)

    def predict_members(
        self, detectors: np.ndarray, members: Iterable[EnsembleMember],
    ) -> dict[EnsembleMember, np.ndarray]:
        """Evaluate each distinct member once for offline ensemble comparison."""
        return {member: self.compile_member(member).decode_batch(detectors)
                for member in dict.fromkeys(members)}


@dataclass
class EnsembleDecoder:
    members: tuple[CompiledDecoder, ...]
    num_detectors: int

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        threshold = len(self.members) // 2 + 1
        ones = self._count_votes(detectors, require_disagreement=False)
        return (ones[:, None] >= threshold).astype(np.uint8)

    def predictions_and_disagreement(self, detectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return majority predictions and the shots with a dissenting vote."""
        threshold = len(self.members) // 2 + 1
        ones = self._count_votes(detectors, require_disagreement=True)
        return (ones[:, None] >= threshold).astype(np.uint8), (ones > 0) & (ones < len(self.members))

    def _count_votes(self, detectors: np.ndarray, *, require_disagreement: bool) -> np.ndarray:
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError("expected a (shots, detectors) array matching the model")
        if not len(detectors):
            return np.empty(0, dtype=np.int32)
        threshold = len(self.members) // 2 + 1
        ones = self.members[0].decode_batch(detectors)[:, 0].astype(np.int32)
        active = None
        for index, member in enumerate(self.members[1:], start=1):
            if index >= threshold:
                previous_ones = ones if active is None else ones[active]
                undecided = (previous_ones < threshold) & (index - previous_ones < threshold)
                if require_disagreement:
                    undecided |= (previous_ones == 0) | (previous_ones == index)
                if not undecided.any():
                    break
                if not undecided.all():
                    active = np.flatnonzero(undecided) if active is None else active[undecided]
            if active is None:
                ones += member.decode_batch(detectors)[:, 0]
            else:
                ones[active] += member.decode_batch(detectors[active])[:, 0]
        return ones

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        if detectors.ndim != 1 or len(detectors) != self.num_detectors:
            raise ValueError("expected a detector vector matching the model")
        threshold = len(self.members) // 2 + 1
        ones = 0
        for count, member in enumerate(self.members, start=1):
            prediction = member.decode_one(detectors)
            ones += int(prediction[0])
            if ones == threshold or count - ones == threshold:
                return prediction
        raise RuntimeError("ensemble vote did not reach a majority")

    def prediction_and_disagreement(self, detectors: np.ndarray) -> tuple[np.ndarray, bool]:
        """Single-shot counterpart to predictions_and_disagreement."""
        if detectors.ndim != 1 or len(detectors) != self.num_detectors:
            raise ValueError("expected a detector vector matching the model")
        threshold = len(self.members) // 2 + 1
        ones = 0
        for count, member in enumerate(self.members, start=1):
            ones += int(member.decode_one(detectors)[0])
            zeros = count - ones
            if ones and zeros and max(ones, zeros) >= threshold:
                return np.array([ones >= threshold], dtype=np.uint8), True
        return np.array([ones >= threshold], dtype=np.uint8), 0 < ones < len(self.members)
