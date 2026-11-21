"""Compile a decoder-only Pauli-axis prior for generated rotated memories.

The bath profile contributes axis ratios conditional on a passing check and
a nonidentity Pauli. The decoder hypothesis retains the generator's total idle
error probability. It uses no physical check flags, timing or erasure rates.

This example uses ``physical`` to sample detectors and ``model`` to compile
the decoder.

    physical = memory_circuit("x", 3, 3, 0.01)
    prior = load_bath_prior("bath-profile.json", channel="clock")
    model = bath_prior_dem(physical, basis="x", distance=3, rounds=3, p=0.01, prior=prior)
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import stim

from .circuits import memory_circuit

BathChannel = Literal["clock", "selected"]
PauliProbabilities = tuple[float, float, float]
_AXES = ("X", "Y", "Z")
_FACTORIZATION_ERROR = "idle Pauli channel cannot be represented by independent X, Y and Z errors"


def _check_channel(channel: BathChannel) -> None:
    if channel not in ("clock", "selected"):
        raise ValueError("bath channel must be 'clock' or 'selected'")


def _probability(value: object) -> float:
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("bath Pauli probabilities must be finite numbers in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class BathPrior:
    """Nonidentity Pauli ratios in X, Y, Z order and their profile channel."""

    channel: BathChannel
    ratios: PauliProbabilities

    def __post_init__(self) -> None:
        _check_channel(self.channel)
        if len(self.ratios) != 3:
            raise ValueError("bath prior requires three ratios in X, Y, Z order")
        x, y, z = map(_probability, self.ratios)
        total = math.fsum((x, y, z))
        if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("bath prior ratios must sum to 1")
        object.__setattr__(self, "ratios", (x / total, y / total, z / total))


def bath_prior_from_profile(
    profile: Mapping[str, object], *, channel: BathChannel = "clock"
) -> BathPrior:
    """Extract and validate the Pauli distribution for one version 1 channel."""
    _check_channel(channel)
    version = profile.get("format_version") if isinstance(profile, Mapping) else None
    if type(version) is not int or version != 1:
        raise ValueError("bath profile format_version must be 1")
    record = profile.get(channel)
    paulis = record.get("conditional_paulis") if isinstance(record, Mapping) else None
    if not isinstance(paulis, Mapping) or set(paulis) != {"I", *_AXES}:
        raise ValueError(f"bath profile channel {channel!r} must contain conditional_paulis for I, X, Y and Z")
    identity, x, y, z = (_probability(paulis[axis]) for axis in ("I", *_AXES))
    if not math.isclose(math.fsum((identity, x, y, z)), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("conditional Pauli probabilities must sum to 1")
    nonidentity = math.fsum((x, y, z))
    if nonidentity == 0:
        raise ValueError(f"bath profile channel {channel!r} has no nonidentity Pauli events")
    return BathPrior(channel, (x / nonidentity, y / nonidentity, z / nonidentity))


def load_bath_prior(path: str | Path, *, channel: BathChannel = "clock") -> BathPrior:
    """Load a bath-initialize JSON export and extract one channel's prior."""
    return bath_prior_from_profile(json.loads(Path(path).read_text(encoding="utf-8-sig")), channel=channel)


def _independent_pauli_errors(paulis: PauliProbabilities) -> PauliProbabilities:
    # A Pauli channel's eigenvalue on X is 1 - 2(pY + pZ). Its independent
    # representation gives the same eigenvalue as (1 - 2qY)(1 - 2qZ).
    x, y, z = paulis
    eigenvalues = (1 - 2 * (y + z), 1 - 2 * (x + z), 1 - 2 * (x + y))
    zeros = eigenvalues.count(0)
    if zeros == 1:
        raise ValueError(_FACTORIZATION_ERROR)
    if zeros == 3:
        return 0.5, 0.5, 0.0
    if zeros == 2:
        # The independent mechanism on the sole nonzero eigenvalue's axis
        # must be fair. The other two factors only constrain their product.
        axis = next(i for i, value in enumerate(eigenvalues) if value != 0)
        probs = [0.0, 0.0, 0.0]
        probs[axis] = 0.5
        probs[(axis + 1) % 3] = (1 - eigenvalues[axis]) / 2
        return probs[0], probs[1], probs[2]

    negatives = sum(value < 0 for value in eigenvalues)
    if negatives % 2:
        raise ValueError(_FACTORIZATION_ERROR)
    probs = []
    for i, error in enumerate(paulis):
        a, b, c = (eigenvalues[(i + offset) % 3] for offset in range(3))
        factor = math.sqrt(abs(b * c / a))
        if negatives:
            if factor > 1 + 8 * math.ulp(1.0):
                raise ValueError(_FACTORIZATION_ERROR)
            signed_factor = -min(factor, 1) if a > 0 else min(factor, 1)
            probs.append((1 - signed_factor) / 2)
        else:
            # Rationalizing 1 - sqrt(b*c/a) avoids erasing tiny error rates
            # when its square root rounds to 1.
            combined = (error + paulis[(i + 1) % 3]) * (error + paulis[(i + 2) % 3])
            gap = error - combined
            if gap < -8 * math.ulp(max(error, combined)):
                raise ValueError(_FACTORIZATION_ERROR)
            probs.append(2 * max(0.0, gap) / (a * (1 + factor)))
    return probs[0], probs[1], probs[2]


def _replace_idle_noise(
    circuit: stim.Circuit, template: stim.Circuit, probs: PauliProbabilities
) -> tuple[stim.Circuit, int]:
    """Align an exact generator output with its idle-only companion."""
    out = stim.Circuit()
    # Stim can merge noiseless initial data and ancilla resets. Align by the
    # tick boundaries and Hadamards that bracket each idle, including its
    # exact targets and probability from the idle-only generator.
    anchors = ("TICK", "H")
    expected_ops = (
        op for op in template
        if isinstance(op, stim.CircuitRepeatBlock) or op.name in (*anchors, "DEPOLARIZE1")
    )
    expected = next(expected_ops, None)
    sites = 0
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if not isinstance(expected, stim.CircuitRepeatBlock) or op.repeat_count != expected.repeat_count:
                raise ValueError("cannot align generator repeat blocks with the idle-noise template")
            body, body_sites = _replace_idle_noise(op.body_copy(), expected.body_copy(), probs)
            out.append(stim.CircuitRepeatBlock(op.repeat_count, body))
            sites += op.repeat_count * body_sites
            expected = next(expected_ops, None)
        elif op == expected:
            if op.name == "DEPOLARIZE1":
                targets = op.targets_copy()
                for axis, probability in zip(_AXES, probs, strict=True):
                    if probability:
                        out.append(f"{axis}_ERROR", targets, probability)
                sites += 1
            else:
                out.append(op)
            expected = next(expected_ops, None)
        elif op.name in anchors:
            raise ValueError("cannot align the generated circuit with the idle-noise template")
        else:
            out.append(op)
    if expected is not None:
        raise ValueError("cannot align the generated circuit with the idle-noise template")
    return out, sites


def bath_prior_circuit(
    circuit: stim.Circuit, *, basis: str, distance: int, rounds: int, p: float, prior: BathPrior,
    expand_isotropic: bool = False,
) -> stim.Circuit:
    """Copy a generated simple_uniform circuit with biased round-start data idles.

    Only the round-start data noise is replaced. Each matching DEPOLARIZE1(p)
    becomes an exact composition with categorical probabilities p times the
    profile's axis ratios. Use the original circuit for sampling.

    An isotropic prior preserves the original circuit by default. Set
    expand_isotropic to use independent Pauli mechanisms as a control.
    """
    if circuit != memory_circuit(basis, distance, rounds, p):
        raise ValueError("bath prior requires the exact generated simple_uniform circuit")
    x, y, z = prior.ratios
    if p == 0 or (x == y == z and not expand_isotropic):
        return circuit.copy()
    try:
        probs = _independent_pauli_errors((p * x, p * y, p * z))
    except ValueError as error:
        raise ValueError(f"{error} at p={p:g} for the {prior.channel!r} bath prior") from error
    template = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=rounds,
        before_round_data_depolarization=p,
    )
    hypothesis, sites = _replace_idle_noise(circuit, template, probs)
    if sites != rounds:
        raise ValueError(f"expected {rounds} generated round-start idle groups, found {sites}")
    return hypothesis


def bath_prior_dem(
    circuit: stim.Circuit, *, basis: str, distance: int, rounds: int, p: float, prior: BathPrior,
    expand_isotropic: bool = False,
) -> stim.DetectorErrorModel:
    """Build a graphlike DEM with correlated components for the decoder hypothesis."""
    hypothesis = bath_prior_circuit(
        circuit, basis=basis, distance=distance, rounds=rounds, p=p, prior=prior,
        expand_isotropic=expand_isotropic,
    )
    return hypothesis.detector_error_model(decompose_errors=True, approximate_disjoint_errors=False)
