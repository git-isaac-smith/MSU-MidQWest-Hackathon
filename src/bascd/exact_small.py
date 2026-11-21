"""Bounded-accuracy maximum-likelihood decoding for small detector models."""

from collections import defaultdict
from dataclasses import dataclass
from math import expm1, fsum, log1p

import numpy as np
import stim


MAX_DETECTORS = 24
_CHUNK_BITS = 16
_CHUNK_SIZE = 1 << _CHUNK_BITS
_FLOAT64 = np.finfo(np.float64)
_UNIT_ROUNDOFF = float(_FLOAT64.eps) / 2
_LIBM_ROUNDOFF = 8 * _UNIT_ROUNDOFF


def _walsh_hadamard(values: np.ndarray) -> None:
    """Apply the unnormalized Walsh–Hadamard transform in bounded scratch space."""
    size = len(values)
    scratch = np.empty(min(size // 2, _CHUNK_SIZE), dtype=values.dtype)
    for group in values.reshape(-1, min(size, 2 * _CHUNK_SIZE)):
        for stage in range(len(group).bit_length() - 1):
            half = 1 << stage
            pairs = group.reshape(-1, 2, half)
            left, right = pairs[:, 0], pairs[:, 1]
            saved = scratch[:left.size].reshape(left.shape)
            np.copyto(saved, left)
            np.add(left, right, out=left)
            np.subtract(saved, right, out=right)

    for stage in range(_CHUNK_BITS + 1, size.bit_length() - 1):
        half = 1 << stage
        for left, right in values.reshape(-1, 2, half):
            for start in range(0, half, _CHUNK_SIZE):
                end = start + _CHUNK_SIZE
                low, high = left[start:end], right[start:end]
                np.copyto(scratch, low)
                np.add(low, high, out=low)
                np.subtract(scratch, high, out=high)


def _zero_half_fault_frequencies(values: np.ndarray, faults: list[tuple[int, bool]]) -> None:
    """Set expm1(log(characteristic)) to -1 where a fair fault cancels it."""
    if not faults:
        return
    width = min(len(values), _CHUNK_SIZE)
    indices = np.arange(width, dtype=np.uint32)
    parity = np.array([index.bit_count() % 2 for index in range(256)], dtype=bool)
    for mask, logical in faults:
        lower_masked = indices & (mask & (width - 1))
        lower_odd = parity[lower_masked & 255] ^ parity[lower_masked >> 8]
        for offset in range(0, len(values), width):
            upper_odd = bool((offset & mask).bit_count() % 2)
            values[offset:offset + width][lower_odd ^ upper_odd ^ logical] = -1


@dataclass(frozen=True)
class ExactSmallDecoder:
    """An offline posterior-sign table with a bound on numerical ML regret.

    `signed_mass_error_bound` bounds the absolute numerical error in each
    syndrome's joint mass difference. `excess_ler_bound` bounds the extra
    logical-error probability over ML, under standard float64 rounding and an
    eight-unit-roundoff allowance for the platform's log1p and expm1.
    """

    num_detectors: int
    estimated_optimal_ler: float
    excess_ler_bound: float
    uncertain_syndromes: int
    signed_mass_error_bound: float
    _negative: np.ndarray
    _detector_offset: int
    _observable_offset: bool

    def _packed(self, detectors: np.ndarray, ndim: int) -> np.ndarray:
        detectors = np.asarray(detectors)
        if detectors.ndim != ndim or detectors.shape[-1] != self.num_detectors:
            raise ValueError(f"expected {self.num_detectors} unpacked detector bits per shot")
        if detectors.dtype != np.bool_:
            if not np.issubdtype(detectors.dtype, np.integer):
                raise ValueError("detectors must contain binary integer or boolean values")
            if detectors.size and (detectors.min() < 0 or detectors.max() > 1):
                raise ValueError("detectors must contain binary integer or boolean values")
        return np.packbits(detectors, axis=-1, bitorder="little")

    def decode_batch(self, detectors: np.ndarray) -> np.ndarray:
        packed = self._packed(detectors, 2)
        indices = np.zeros((len(packed), 4), dtype=np.uint8)
        indices[:, :packed.shape[1]] = packed
        indices = indices.view("<u4").ravel() ^ self._detector_offset
        return (self._negative[indices] ^ self._observable_offset)[:, None]

    def decode_one(self, detectors: np.ndarray) -> np.ndarray:
        index = int.from_bytes(self._packed(detectors, 1), "little") ^ self._detector_offset
        return np.array([self._negative[index] ^ self._observable_offset])


def compile_exact_small(model: stim.DetectorErrorModel) -> ExactSmallDecoder:
    """Compile from an undecomposed DEM with at most 24 detectors and one observable.

    Each DEM error is an independent Bernoulli mechanism over its full detector
    and logical target sets. The inverse transform gives the signed joint mass
    P(syndrome, L=0) - P(syndrome, L=1), so its sign is the ML decision.
    """
    detectors = model.num_detectors
    if detectors > MAX_DETECTORS or model.num_observables != 1:
        raise ValueError(f"small ML requires at most {MAX_DETECTORS} detectors and one observable")

    log_coefficients = defaultdict(list)
    log_weights = []
    fair_faults = []
    detector_offset, observable_offset = 0, False
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        mask, logical = 0, False
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << target.val
            elif target.is_logical_observable_id():
                logical = not logical
            else:
                raise ValueError("small ML requires an undecomposed detector error model")

        probability = instruction.args_copy()[0]
        if probability > 0.5:
            detector_offset ^= mask
            observable_offset ^= logical
            probability = 1 - probability
        if probability == 0 or (mask == 0 and not logical):
            continue
        if probability == 0.5:
            fair_faults.append((mask, logical))
            continue
        # A fault contributes log(1-2p) only at frequencies with odd joint parity.
        weight = 0.5 * log1p(-2 * probability)
        log_weights.append(-weight)
        log_coefficients[0].append(weight)
        log_coefficients[mask].append(weight if logical else -weight)

    values = np.zeros(1 << detectors, dtype=np.float64)
    for mask, terms in log_coefficients.items():
        values[mask] = fsum(terms)

    _walsh_hadamard(values)
    np.minimum(values, 0, out=values)
    # Remove the known probability atom before inversion to reduce cancellation.
    np.expm1(values, out=values)
    _zero_half_fault_frequencies(values, fair_faults)
    mean_spectrum_magnitude = -float(values.mean())
    _walsh_hadamard(values)
    values /= len(values)
    zero_residual = float(values[0])
    # Once deterministic offsets are removed, all Fourier coefficients are
    # nonnegative, so the zero syndrome's signed mass is nonnegative. Keep
    # its residual separate to avoid subtracting nearly equal numbers in LER.
    values[0] = 0

    # The forward butterfly propagates absolute input errors with gain at most
    # the L1 norm of its coefficients. Exponentiation on nonpositive inputs is
    # 1-Lipschitz. Inverse butterfly errors are scaled by the mean absolute
    # spectrum because the inverse includes division by the table size.
    unit = _UNIT_ROUNDOFF
    butterfly_error = detectors * unit / (1 - detectors * unit)
    log_norm = 2 * fsum(log_weights)
    log_error = (butterfly_error + _LIBM_ROUNDOFF + 4 * unit) * log_norm
    spectrum_bound = 1 if fair_faults else min(1, -expm1(-log_norm) + log_error)
    expm1_error = _LIBM_ROUNDOFF * spectrum_bound
    inverse_error = butterfly_error * mean_spectrum_magnitude * (1 + 2 * len(values) * unit)
    mass_error = (log_error + expm1_error + inverse_error) * (1 + 16 * unit)
    if log_weights or fair_faults:
        mass_error += (10 * len(log_weights) + 10) * float(_FLOAT64.smallest_subnormal)

    negative = np.empty(len(values), dtype=bool)
    np.less(values, 0, out=negative)
    negative.flags.writeable = False
    mass, regret, uncertain = [], [], 0
    for start in range(0, len(values), _CHUNK_SIZE):
        magnitudes = np.abs(values[start:start + _CHUNK_SIZE])
        mass.append(float(magnitudes.sum()))
        uncertainty = mass_error - magnitudes
        if start == 0:
            uncertainty[0] = 0
        uncertain += int(np.count_nonzero(uncertainty > 0))
        np.maximum(uncertainty, 0, out=uncertainty)
        regret.append(float(uncertainty.sum()))

    return ExactSmallDecoder(
        num_detectors=detectors,
        estimated_optimal_ler=max(0.0, (-zero_residual - fsum(mass)) / 2),
        excess_ler_bound=min(1.0, fsum(regret) * (1 + 2 * (_CHUNK_SIZE + len(regret)) * unit)),
        uncertain_syndromes=uncertain,
        signed_mass_error_bound=mass_error,
        _negative=negative,
        _detector_offset=detector_offset,
        _observable_offset=observable_offset,
    )
