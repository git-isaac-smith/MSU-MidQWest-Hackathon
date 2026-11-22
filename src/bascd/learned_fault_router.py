"""Learn when correlated matching recovers logical errors from packed syndromes."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pymatching
from scipy.optimize import minimize

from .css_routed import _PACKED, observable_detector_mask


_BYTE_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little")
_MODEL_VERSION = 3
_LANE_LIMIT = 8
_COUNT_LIMIT = 63
_LANE_COUNTS = _BYTE_BITS.sum(axis=1, dtype=np.uint8)


def _model_digest(model):
    return hashlib.sha256(str(model).encode()).hexdigest()


def _compile_projections(circuit, model, temporal, parity):
    matcher = pymatching.Matching.from_detector_error_model(model)
    width = (model.num_detectors + 7) // 8
    mask = observable_detector_mask(matcher, model.num_detectors)
    relevant = np.unpackbits(mask, bitorder="little").astype(bool)
    coordinates = circuit.get_detector_coordinates()
    logical_boundary = [u for u, v, attrs in matcher.edges() if v is None and 0 in attrs["fault_ids"]]
    lanes = np.zeros((width, 8), dtype=np.uint64)
    parity_flags = np.zeros((width, 8), dtype=np.uint8)
    time_count = 1
    lane_count = 0
    if logical_boundary and all(len(coordinates.get(i, ())) >= 2 for i in range(model.num_detectors)):
        spatial = np.asarray([coordinates[i][:2] for i in range(model.num_detectors)], dtype=float)
        boundary = spatial[logical_boundary]
        fixed_axes = np.flatnonzero(np.ptp(boundary, axis=0) == 0)
        if len(fixed_axes) == 1:
            axis = fixed_axes[0]
            selected = np.flatnonzero(relevant)
            _, ranks = np.unique(spatial[selected, axis], return_inverse=True)
            if len(ranks) and ranks.max() < _LANE_LIMIT:
                if boundary[0, axis] == np.max(spatial[selected, axis]):
                    ranks = ranks.max() - ranks
                lane_count = int(ranks.max()) + 1
                parity_flags.ravel()[selected] = np.left_shift(1, ranks).astype(np.uint8)
                position = ranks
                if temporal and all(len(coordinates[i]) >= 3 for i in selected):
                    _, times = np.unique([coordinates[i][2] for i in selected], return_inverse=True)
                    if 1 < times.max() + 1 <= 8:
                        time_count = int(times.max()) + 1
                        position = position + 8 * times
                lanes.ravel()[selected] = np.left_shift(np.uint64(1), position.astype(np.uint64))
    valid = np.arange(width * 8) < model.num_detectors
    # Both independent counts fit in one summed uint32 lane.
    flags = np.where(relevant, np.uint32(1 << 16), np.uint32(1)) * valid
    activity_lookup = flags.reshape(width, 8) @ _BYTE_BITS.T.astype(np.uint32)
    lane_lookup = np.bitwise_or.reduce(lanes[:, None, :] * _BYTE_BITS[None, :, :], axis=2)
    if time_count == 1:
        lane_lookup = lane_lookup.astype(np.uint8)
    parity_lookup = None
    if parity and lane_count:
        parity_lookup = np.bitwise_xor.reduce(parity_flags[:, None, :] * _BYTE_BITS[None, :, :], axis=2)
    return activity_lookup, lane_lookup, parity_lookup, lane_count, time_count


def _activity_fields(packed, activity_lookup):
    columns = np.arange(packed.shape[1])
    counts = activity_lookup[columns, packed].sum(axis=1, dtype=np.uint32)
    return (counts >> 16).astype(np.intp), (counts & 65535).astype(np.intp)


def _lane_fields(packed, lane_lookup, time_count):
    projection = np.bitwise_or.reduce(lane_lookup[np.arange(packed.shape[1]), packed], axis=1)
    if time_count == 1:
        return projection, np.zeros(len(projection), dtype=np.uint8)
    times = projection.astype("<u8", copy=False).view(np.uint8).reshape(-1, 8)[:, :time_count]
    lanes = np.bitwise_or.reduce(times, axis=1)
    adjacent = times[:, :-1] | times[:, 1:]
    return lanes, _LANE_COUNTS[adjacent].max(axis=1)


def _parity_field(packed, parity_lookup, lane_count):
    if parity_lookup is None:
        return np.zeros(len(packed), dtype=np.uint8)
    lane_parities = np.bitwise_xor.reduce(parity_lookup[np.arange(packed.shape[1]), packed], axis=1)
    # The two boundary choices complement crossings of the lane cuts.
    prefix = lane_parities ^ (lane_parities << 1)
    prefix ^= prefix << 2
    prefix ^= prefix << 4
    crossings = _LANE_COUNTS[prefix & ((1 << lane_count) - 1)]
    return np.minimum(crossings, lane_count + 1 - crossings)


def _fields(packed, activity_lookup, lane_lookup, parity_lookup, lane_count, time_count):
    return (*_activity_fields(packed, activity_lookup), *_lane_fields(packed, lane_lookup, time_count),
            _parity_field(packed, parity_lookup, lane_count))


def _basic_features(relevant, other, lanes, lane_count, adjacent_occupancy=None, parity=None):
    occupancy = _LANE_COUNTS[lanes]
    if lane_count:
        weights = (1 << np.arange(lane_count)).astype(np.uint8)
        left = np.argmax((lanes[:, None] & weights) != 0, axis=1)
        right = lane_count - 1 - np.argmax((lanes[:, None] & weights[::-1]) != 0, axis=1)
        span = np.where(lanes != 0, right - left + 1, 0)
        endpoint = ((lanes & 1) != 0).astype(np.float64) + ((lanes & weights[-1]) != 0)
    else:
        span = endpoint = np.zeros(len(lanes))
    features = [relevant, other, occupancy, span, endpoint]
    if adjacent_occupancy is not None:
        features.append(adjacent_occupancy)
    if parity is not None:
        features.append(parity)
    return np.column_stack(features)


def _terms(basic, center, scale):
    standardized = np.clip((basic - center) / scale, -4, 4)
    left, right = np.triu_indices(basic.shape[1])
    return np.column_stack([np.ones(len(basic)), standardized, standardized[:, left] * standardized[:, right]])


def _learn_rates(features, exposures, outcomes, base_count=5):
    """Fit separate regularized rare-event rates for recoveries and regressions."""
    penalty = np.r_[0.0, np.full(base_count, 3.0), np.full(features.shape[1] - 1 - base_count, 12.0)]
    coefficients = []
    for counts in outcomes:
        if not np.any(counts):
            coefficients.append(np.r_[-40.0, np.zeros(features.shape[1] - 1)])
            continue
        observed = counts @ features
        initial = np.r_[np.log(counts.sum() / exposures.sum()), np.zeros(features.shape[1] - 1)]

        def objective(beta):
            log_rate = features @ beta
            expected = exposures * np.exp(np.minimum(log_rate, 50))
            regularizer = penalty * beta
            return (expected.sum() - observed @ beta + .5 * regularizer @ beta,
                    features.T @ expected - observed + regularizer)

        result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                          options=dict(maxiter=400, ftol=1e-10))
        coefficients.append(result.x)
    return np.asarray(coefficients)


def _as_predictions(values, shots):
    values = np.asarray(values)
    if values.shape not in ((shots,), (shots, 1)) or values.dtype.kind not in "biu" or np.any((values < 0) | (values > 1)):
        raise ValueError("expected one binary observable per shot")
    return values.reshape(shots)


def _validate_records(packed, width):
    if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != width:
        raise ValueError("expected little-endian packed detector rows")


def _histogram_fields(packed, projections, gain):
    _, _, parity_lookup, lane_count, time_count = projections
    fields = np.concatenate([
        np.column_stack(_fields(packed[start:start + 8192], *projections))
        for start in range(0, len(packed), 8192)
    ])
    # Cap extreme tails to bound the frozen score table and pool rare cases.
    limits = np.quantile(fields[:, :2], .999, axis=0, method="higher").astype(np.intp)
    limits = np.minimum(limits, _COUNT_LIMIT)
    fields[:, :2] = np.minimum(fields[:, :2], limits)
    shape = (limits[0] + 1, limits[1] + 1, 1 << lane_count, lane_count + 1 if time_count > 1 else 1,
             (lane_count + 1) // 2 + 1 if parity_lookup is not None else 1)
    keys = np.ravel_multi_index(fields.T, shape)
    exposure = np.bincount(keys, minlength=int(np.prod(shape)))
    occupied = np.flatnonzero(exposure)
    outcomes = [np.bincount(keys[gain == sign], minlength=len(exposure))[occupied] for sign in (1, -1)]
    return shape, occupied, exposure[occupied], outcomes


def _features_for_bins(indices, shape):
    lane_count = (shape[2] - 1).bit_length()
    relevant, other, lanes, adjacent, parity = np.unravel_index(indices, shape)
    return _basic_features(relevant, other, lanes, lane_count,
                           adjacent if shape[3] > 1 else None, parity if shape[4] > 1 else None)


def _table_scores(shape, coefficients, center, scale):
    scores = np.empty(int(np.prod(shape)), dtype=np.float32)
    for start in range(0, len(scores), 8192):
        stop = min(start + 8192, len(scores))
        basic = _features_for_bins(np.arange(start, stop), shape)
        features = _terms(basic, center, scale)
        rates = np.exp(np.minimum(features @ coefficients.T, 0))
        scores[start:stop] = rates[:, 0] - rates[:, 1]
    return scores.reshape(shape)


class LearnedFaultRouter:
    """A frozen packed-syndrome score table learned from paired decoder outcomes."""

    def __init__(self, model_digest, activity_lookup, lane_lookup, time_count, scores, cutoff,
                 parity_lookup=None):
        self.model_digest = model_digest
        self.activity_lookup = activity_lookup
        self.lane_lookup = lane_lookup
        self.parity_lookup = parity_lookup
        self.time_count = time_count
        self.lane_count = (scores.shape[2] - 1).bit_length()
        self.scores = scores
        self.cutoff = float(cutoff)
        route = scores > cutoff
        feasible = np.ones((scores.shape[0], scores.shape[2], scores.shape[3]), dtype=bool)
        if scores.shape[2] > 1:
            counts = np.arange(scores.shape[0])[:, None]
            occupied = _LANE_COUNTS[:scores.shape[2]]
            feasible_count = ((counts >= occupied) | (counts == scores.shape[0] - 1))
            feasible_count &= (counts == 0) == (occupied == 0)
            if scores.shape[0] == 1:
                feasible_count[0] = True
            feasible &= feasible_count[..., None]
            if scores.shape[3] > 1:
                adjacent = np.arange(scores.shape[3])
                feasible &= (adjacent <= occupied[:, None]) & ((adjacent == 0) == (occupied[:, None] == 0))
        self._always = np.all(route | ~feasible[:, None, :, :, None], axis=(2, 3, 4))
        self._ambiguous = np.any(route & feasible[:, None, :, :, None], axis=(2, 3, 4)) & ~self._always

    @classmethod
    def fit(cls, circuit, model, packed, ordinary, correlated, truth, *, fraction=.06, temporal=True,
            parity=False):
        """Fit and calibrate a routing cutoff on caller-supplied training records."""
        if (circuit.num_detectors != model.num_detectors or model.num_observables != 1
                or not 1 <= model.num_detectors < 65536 or not 0 <= fraction <= 1
                or not len(packed)):
            raise ValueError("requires a matching circuit and one observable, training records and a valid fraction")
        _validate_records(packed, (model.num_detectors + 7) // 8)
        ordinary = _as_predictions(ordinary, len(packed))
        correlated = _as_predictions(correlated, len(packed))
        truth = _as_predictions(truth, len(packed))
        gain = (ordinary != truth).astype(np.int8) - (correlated != truth)
        projections = _compile_projections(circuit, model, temporal, parity)
        activity_lookup, lane_lookup, parity_lookup, _, time_count = projections
        shape, occupied, exposure, outcomes = _histogram_fields(packed, projections, gain)
        original = _features_for_bins(occupied, shape)
        center = np.average(original, axis=0, weights=exposure)
        scale = np.maximum(.5, np.sqrt(np.average((original - center)**2, axis=0, weights=exposure)))
        features = _terms(original, center, scale)
        coefficients = _learn_rates(features, exposure, outcomes, base_count=original.shape[1])
        scores = _table_scores(shape, coefficients, center, scale)
        calibration_scores = scores.ravel()[occupied]
        order = np.argsort(calibration_scores, kind="stable")
        if fraction == 0:
            cutoff = np.inf
        elif fraction == 1:
            cutoff = -np.inf
        else:
            target = min(len(packed), int(np.floor((1 - fraction) * len(packed))) + 1)
            index = np.searchsorted(np.cumsum(exposure[order]), target)
            cutoff = max(0, calibration_scores[order[index]])
        return cls(_model_digest(model), activity_lookup, lane_lookup, time_count, scores, cutoff, parity_lookup)

    @classmethod
    def fit_sampled(cls, circuit, model, *, seeds, shots_per_seed, fraction=.06, temporal=True, parity=False):
        """Train from independent Stim sampler seeds using both matching branches."""
        seeds = tuple(seeds)
        if not seeds or len(set(seeds)) != len(seeds) or shots_per_seed < 1:
            raise ValueError("requires unique training seeds and a positive sample size")
        ordinary = pymatching.Matching.from_detector_error_model(model)
        correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        mask = observable_detector_mask(ordinary, model.num_detectors)
        records = []
        for seed in seeds:
            packed, truth = circuit.compile_detector_sampler(seed=seed).sample(
                shots_per_seed, separate_observables=True, bit_packed=True)
            easy = ordinary.decode_batch(packed & mask, **_PACKED)
            hard = correlated.decode_batch(packed, enable_correlations=True, **_PACKED)
            records.append((packed, easy, hard, truth))
        return cls.fit(circuit, model, *(np.concatenate(group) for group in zip(*records)),
                       fraction=fraction, temporal=temporal, parity=parity)

    def score_batch(self, packed):
        """Estimate the error reduction of correlated matching on each packed row."""
        _validate_records(packed, len(self.activity_lookup))
        relevant, other, lanes, adjacent, parity = _fields(
            packed, self.activity_lookup, self.lane_lookup, self.parity_lookup, self.lane_count, self.time_count)
        return self.scores[np.minimum(relevant, self.scores.shape[0] - 1),
                           np.minimum(other, self.scores.shape[1] - 1), lanes, adjacent, parity]

    def route_batch(self, packed):
        """Return a Boolean mask selecting full correlated matching."""
        _validate_records(packed, len(self.activity_lookup))
        relevant, other = _activity_fields(packed, self.activity_lookup)
        np.minimum(relevant, self.scores.shape[0] - 1, out=relevant)
        np.minimum(other, self.scores.shape[1] - 1, out=other)
        route = self._always[relevant, other]
        ambiguous = self._ambiguous[relevant, other]
        if np.any(ambiguous):
            selected = packed[ambiguous]
            lanes, adjacent = _lane_fields(selected, self.lane_lookup, self.time_count)
            parity = _parity_field(selected, self.parity_lookup, self.lane_count)
            route[ambiguous] = self.scores[relevant[ambiguous], other[ambiguous], lanes, adjacent, parity] > self.cutoff
        return route

    def save(self, path):
        """Write the numerical policy in NumPy's pickle-free archive format."""
        with open(path, "wb") as stream:
            np.savez_compressed(stream, version=_MODEL_VERSION, model_digest=self.model_digest,
                                activity_lookup=self.activity_lookup, lane_lookup=self.lane_lookup,
                                time_count=self.time_count, scores=self.scores, cutoff=self.cutoff,
                                parity_lookup=(self.parity_lookup if self.parity_lookup is not None else
                                               np.empty(0, dtype=np.uint8)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            version = archive["version"].item()
            if version not in (1, 2, _MODEL_VERSION):
                raise ValueError("unsupported routing policy version")
            time_count = archive["time_count"].item() if version > 1 else 1
            scores = archive["scores"]
            if version < 2:
                scores = scores[..., None]
            if version < 3:
                scores = scores[..., None]
            parity_lookup = archive["parity_lookup"] if version >= 3 else None
            if parity_lookup is not None and not parity_lookup.size:
                parity_lookup = None
            return cls(archive["model_digest"].item(), archive["activity_lookup"], archive["lane_lookup"],
                       time_count, scores, archive["cutoff"].item(), parity_lookup)


class LearnedFaultRoutedDecoder:
    """Run masked ordinary matching, with learned escalation to full correlated matching."""

    def __init__(self, model, policy):
        if policy.model_digest != _model_digest(model):
            raise ValueError("routing policy does not match the detector error model")
        self.policy = policy
        self.ordinary = pymatching.Matching.from_detector_error_model(model)
        self.correlated = pymatching.Matching.from_detector_error_model(model, enable_correlations=True)
        self.mask = observable_detector_mask(self.ordinary, model.num_detectors)

    def decode_batch(self, packed):
        escalate = self.policy.route_batch(packed)
        easy = packed & self.mask
        easy[escalate] = 0
        prediction = self.ordinary.decode_batch(easy, **_PACKED)
        if np.any(escalate):
            prediction[escalate] = self.correlated.decode_batch(packed[escalate], enable_correlations=True, **_PACKED)
        return prediction


def main():
    """Train a standalone policy for the selected circuit and Stim seed split."""
    from .challenge_variants import VARIANT_NAMES, generate_variant_circuit

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit and save a detector-only routing policy")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--basis", choices=("x", "z"), default="z")
    train.add_argument("--distance", type=int, required=True)
    train.add_argument("--rounds", type=int)
    train.add_argument("--p", type=float, required=True)
    train.add_argument("--variant", choices=VARIANT_NAMES, default="ordinary")
    train.add_argument("--seeds", nargs="+", type=int, required=True)
    train.add_argument("--shots-per-seed", type=int, required=True)
    train.add_argument("--fraction", type=float, default=.06)
    train.add_argument("--no-temporal", action="store_true")
    train.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    try:
        circuit = generate_variant_circuit(args.variant, basis=args.basis, distance=args.distance,
                                           rounds=args.rounds if args.rounds is not None else args.distance,
                                           p=args.p)
        model = circuit.detector_error_model(decompose_errors=True)
        policy = LearnedFaultRouter.fit_sampled(
            circuit, model, seeds=args.seeds, shots_per_seed=args.shots_per_seed, fraction=args.fraction,
            temporal=not args.no_temporal, parity=args.parity)
        policy.save(args.out)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
