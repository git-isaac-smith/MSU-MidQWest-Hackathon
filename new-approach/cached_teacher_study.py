"""Archive-trained detector-only posterior cache for the five-round Z-memory study."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import stim
from scipy.stats import binomtest

from bascd.stats import paired_ler_difference, wilson95
from full_map_decoder import FullMapDecoder, MAP_TIE_TOLERANCE
from logical_reference import build_reference
from native_posterior import native_library_path


BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / 'results' / 'full-map-01'
OUT = BASE / 'results' / 'cached-teacher-01'
BLOCK = 4096
WORD_MASK = np.uint64(0xfff)
DECISION = np.uint64(1 << 63)
EMPTY = np.uint64(1 << 62)
PHYSICAL = (5, 0.008, 0.04)
PILOT_SEED = 84597800
TEST_SEEDS = (84598000, 84598001, 84598002)
ROUTE_COSTS = (29.55, 26.47, 23.88)
TIMING_BATCHES = 9
TIMING_SHOTS = 49_152


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, values):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _save_npz(path, **arrays):
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _bucket(keys, mask):
    mixed = (keys ^ (keys >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    mixed = (mixed ^ (mixed >> 27)) * np.uint64(0x94d049bb133111eb)
    return (mixed ^ (mixed >> 31)) & mask


class PackedHistories:
    """Encode all posterior-relevant checks in an exact 60-bit key."""

    def __init__(self, reference):
        times = reference._detector_times.reshape(6, 12)
        columns = reference._detector_checks.reshape(6, 12)
        if (not np.array_equal(times, np.repeat(np.arange(6)[:, None], 12, axis=1)) or
                not np.all(columns[1:5] == np.arange(12)) or
                not np.array_equal(columns[0], columns[-1]) or
                not np.array_equal(np.sort(columns[0]), np.arange(12))):
            raise ValueError('Unexpected packed detector order for the archived circuit')
        values = np.arange(4096, dtype=np.uint64)
        self.outer = np.zeros_like(values)
        for start, end in enumerate(columns[0]):
            self.outer |= ((values >> start) & 1) << np.uint64(end)

    def keys(self, packed, reverse):
        if packed.strides[1] != 1:
            packed = np.ascontiguousarray(packed)
        low = packed[:, :8].view('<u8').ravel()
        first = self.outer[low & WORD_MASK]
        second = (low >> 12) & WORD_MASK
        third = (low >> 24) & WORD_MASK
        fourth = (low >> 36) & WORD_MASK
        fifth = (low >> 48) & WORD_MASK
        last = self.outer[(low >> 60) | (packed[:, 8].astype(np.uint64) << 4)]
        # Terminal = XOR of all six detector differences. The fifth noisy
        # readout likelihood cancels from the posterior given this terminal.
        end_difference = fifth ^ last
        terminal = first ^ second ^ third ^ fourth ^ end_difference
        history = first | (second << 12) | (third << 24) | (fourth << 36)
        if reverse:
            # Reversing the five i.i.d. XOR transitions preserves final logical
            # parity. Relative to terminal, the first reversed noisy check is
            # the XOR of the last two detector differences.
            reversed_history = end_difference | (fourth << 12) | (third << 24) | (second << 36)
            history = np.minimum(history, reversed_history)
        return history | (terminal << 48)


class CachedTeacher:
    """Use the full posterior's decision on a verified hit, and MWPM otherwise."""

    def __init__(self, reference, training, *, reverse, table_bits, route_cost=None):
        self.reference = reference
        self.history = PackedHistories(reference)
        self.reverse = reverse
        self.route_cost = route_cost
        self.posterior = FullMapDecoder(PHYSICAL, 'native') if route_cost is not None else None
        self.mask = np.uint64((1 << table_bits) - 1)
        records, labels = training
        keys = self.history.keys(records, self.reverse)
        order = np.argsort(keys, kind='stable')
        keys, labels = keys[order], labels[order]
        groups = np.r_[0, np.flatnonzero(keys[:-1] != keys[1:]) + 1]
        decisions = np.minimum.reduceat(labels, groups)
        if np.any(decisions != np.maximum.reduceat(labels, groups)):
            raise ValueError('Archived full-posterior predictions disagree for equivalent records')
        keys = keys[groups]
        counts = np.diff(groups, append=len(labels))
        ranked = np.argsort(counts, kind='stable')
        self.table = np.full(1 << table_bits, EMPTY, dtype=np.uint64)
        slots = _bucket(keys[ranked], self.mask)
        winners = np.full(len(self.table), -1, dtype=np.int32)
        np.maximum.at(winners, slots, np.arange(len(ranked), dtype=np.int32))
        occupied = winners >= 0
        retained = ranked[winners[occupied]]
        self.table[occupied] = keys[retained] | (decisions[retained].astype(np.uint64) << 63)
        self.retained = len(retained)
        self.unique = len(keys)
        self.table.flags.writeable = False

    def mwpm(self, packed):
        return self.reference._mwpm.decode_batch(
            packed, bit_packed_shots=True, bit_packed_predictions=True)[:, 0]

    def _lookup(self, packed):
        keys = self.history.keys(packed, self.reverse)
        values = self.table[_bucket(keys, self.mask)]
        return (values & ~DECISION) == keys, (values >> 63).astype(np.uint8)

    def decode(self, packed, *, diagnostics=False):
        if packed.ndim != 2 or packed.shape[1] != 9 or packed.dtype != np.uint8:
            raise ValueError('Expected nine little-endian packed detector bytes per shot')
        hits, prediction = self._lookup(packed)
        misses = ~hits
        routed = np.zeros(len(packed), dtype=bool) if diagnostics else None
        if np.any(misses):
            records = packed[misses]
            if self.posterior is None:
                prediction[misses] = self.mwpm(records)
            else:
                decoded, cost = self.reference._mwpm.decode_batch(
                    records, bit_packed_shots=True, bit_packed_predictions=True, return_weights=True)
                hard = cost >= self.route_cost
                if np.any(hard):
                    unpacked = np.unpackbits(records[hard], axis=1, bitorder='little')
                    probability = self.posterior.probabilities(unpacked)
                    exact = np.where(abs(probability - .5) <= MAP_TIE_TOLERANCE,
                                     decoded[hard, 0], probability > .5)
                    decoded[hard, 0] = exact
                prediction[misses] = decoded[:, 0]
                if diagnostics:
                    routed[misses] = hard
        return (prediction, hits, routed) if diagnostics else prediction


def _teacher(archive):
    plan_path = archive / 'plan.json'
    plan = _read_json(plan_path)
    if tuple(plan['physical']) != PHYSICAL or plan['rounds'] != 5:
        raise ValueError('Wrong physical model for teacher archive')
    provenance = {name: _sha(archive / name) for name in ('plan.json', 'benchmark.stim')}
    records, labels = [], []
    for seed in plan['seeds']:
        sample, posterior = f'sample-{seed}.npz', f'posterior-{seed}.npz'
        provenance.update({name: _sha(archive / name) for name in (sample, posterior)})
        with np.load(archive / sample, allow_pickle=False) as public, np.load(
                archive / posterior, allow_pickle=False) as teacher:
            p = teacher['probability']
            d = teacher['full_map']
            packed = public['detectors']
            if (packed.shape != (plan['shots_per_seed'], 9) or packed.dtype != np.uint8 or
                    p.shape != (len(packed),) or d.shape != p.shape or d.dtype != np.bool_ or
                    not np.all(np.isfinite(p) & (p >= 0) & (p <= 1))):
                raise ValueError('Unexpected teacher archive dimensions or values')
            usable = abs(p - .5) > MAP_TIE_TOLERANCE
            if np.any(d[usable] != (p[usable] > .5)):
                raise ValueError('Teacher labels disagree with archived posteriors')
            records.append(packed[usable])
            labels.append(d[usable])
    return (np.concatenate(records), np.concatenate(labels)), provenance


def _sample(circuit, seed, shots):
    sampler = circuit.compile_detector_sampler(seed=seed)
    packed, truth = sampler.sample(shots, bit_packed=True, separate_observables=True)
    return np.ascontiguousarray(packed), truth[:, 0]


def _batched(decode, packed):
    return np.concatenate([decode(packed[i:i + BLOCK]) for i in range(0, len(packed), BLOCK)])


def _time(candidate, packed, passes, distinct=False):
    functions = {'mwpm': candidate.mwpm, 'cached_teacher': candidate.decode}
    for fn in functions.values():
        fn(packed[:BLOCK])
    times = {key: [] for key in functions}
    order = tuple(functions)
    batches = np.array_split(packed[BLOCK:], passes) if distinct else [packed[BLOCK:]] * passes
    for idx in range(passes):
        for name in order if idx % 2 == 0 else order[::-1]:
            start = perf_counter()
            _batched(functions[name], batches[idx])
            times[name].append(perf_counter() - start)
    medians = {key: float(np.median(values)) for key, values in times.items()}
    pair_speedup = np.asarray(times['mwpm']) / times['cached_teacher']
    replicates = np.random.default_rng(701).choice(pair_speedup, size=(10_000, len(pair_speedup)))
    return dict(shots_per_pass=len(batches[0]), batch_size=BLOCK, seconds=times,
                independent_timing_batches=distinct,
                median_seconds=medians, speedup=medians['mwpm'] / medians['cached_teacher'],
                median_paired_speedup_ci95=np.quantile(np.median(replicates, axis=1), [.025, .975]).tolist())


def _paired(seed, truth, reference, candidate, hits, routed=None):
    base, test = reference != truth, candidate != truth
    return dict(seed=seed, shots=len(truth), hits=int(hits.sum()), routed=int(routed.sum()) if routed is not None else 0,
                mwpm_errors=int(base.sum()),
                cached_teacher_errors=int(test.sum()), recovered=int((base & ~test).sum()),
                lost=int((~base & test).sum()))


def _summary(rows):
    shots, recovered, lost = (sum(row[key] for row in rows) for key in ('shots', 'recovered', 'lost'))
    stats = paired_ler_difference([row['shots'] for row in rows], [row['recovered'] for row in rows],
                                  [row['lost'] for row in rows])
    if stats['approximate_95_ci'] is None:
        gain_ci = binomtest(recovered, shots).proportion_ci(confidence_level=.975, method='exact')
        loss_ci = binomtest(lost, shots).proportion_ci(confidence_level=.975, method='exact')
        bounds = loss_ci.low - gain_ci.high, loss_ci.high - gain_ci.low
        method = 'Conservative exact paired marginal intervals'
    else:
        bounds = stats['approximate_95_ci']
        method = 'Stratified paired normal approximation'
    error_rates = {}
    for name in ('mwpm', 'cached_teacher'):
        errors = sum(row[f'{name}_errors'] for row in rows)
        error_rates[name] = dict(errors=errors, ler=errors / shots, wilson95=wilson95(errors, shots))
    return dict(shots=shots, error_rates=error_rates, hit_rate=sum(row['hits'] for row in rows) / shots,
                recovered=recovered, lost=lost, ler_difference=stats['ler_difference'], paired_ci95=bounds,
                paired_ci_method=method, mcnemar_p=binomtest(recovered, recovered + lost).pvalue
                if recovered + lost else 1.0, routed_rate=sum(row['routed'] for row in rows) / shots)


def run(args):
    ref = build_reference(*PHYSICAL)
    archived_circuit = stim.Circuit.from_file(args.archive / 'benchmark.stim')
    if ref.circuit != archived_circuit:
        raise ValueError('Archive circuit does not match the decoder observation map')
    start = perf_counter()
    training, hashes = _teacher(args.archive)
    archive_load_seconds = perf_counter() - start
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / 'plan.json'
    if args.phase == 'pilot':
        if config.exists():
            raise ValueError('Pilot selection is closed after the plan is frozen')
        pilot_input, pilot_truth = _sample(ref.circuit, PILOT_SEED, 40_960)
        pilots = []
        for reverse in (False, True):
            for table_bits in (19, 21):
                candidate = CachedTeacher(ref, training, reverse=reverse, table_bits=table_bits)
                hits, _ = candidate._lookup(pilot_input)
                counts = _paired(PILOT_SEED, pilot_truth, _batched(candidate.mwpm, pilot_input),
                                 _batched(candidate.decode, pilot_input), hits)
                pilots.append(dict(reverse=reverse, table_bits=table_bits, counts=counts,
                                   timing=_time(candidate, pilot_input, 5)))
        _write_json(args.out / 'pilot.json', pilots)
        for pilot in pilots:
            print(pilot['reverse'], pilot['table_bits'], pilot['counts'], pilot['timing']['speedup'])
        return
    if args.phase == 'route-pilot':
        if config.exists():
            raise ValueError('Pilot selection is closed after the plan is frozen')
        archive_seed = _read_json(args.archive / 'plan.json')['seeds'][-1]
        with np.load(args.archive / f'sample-{archive_seed}.npz', allow_pickle=False) as archived, np.load(
                args.archive / f'posterior-{archive_seed}.npz', allow_pickle=False) as teacher:
            dev = archived['detectors']
            expected_base, ideal, prob = archived['mwpm'], teacher['full_map'], teacher['probability']
        shard_size = np.count_nonzero(abs(prob - .5) > MAP_TIE_TOLERANCE)
        development_training = tuple(values[:-shard_size] for values in training)
        probe = CachedTeacher(ref, development_training, reverse=True, table_bits=21)
        hits, _ = probe._lookup(dev)
        _, cost = ref._mwpm.decode_batch(dev, bit_packed_shots=True, bit_packed_predictions=True,
                                        return_weights=True)
        conditional_gain = np.where(ideal != expected_base, abs(2 * prob - 1), 0)
        timing_input, _ = _sample(ref.circuit, PILOT_SEED + 1, 40_960)
        entries = []
        for threshold in ROUTE_COSTS:
            routed = ~hits & (cost >= threshold)
            candidate = CachedTeacher(ref, training, reverse=True, table_bits=21, route_cost=threshold)
            entry = dict(threshold=threshold, archive_seed=archive_seed, archive_shots=len(dev),
                         archived_fold_hit_rate=float(hits.mean()), archived_fold_routed=int(routed.sum()),
                         model_expected_recoveries=float(conditional_gain[routed].sum()),
                         timing=_time(candidate, timing_input, 5))
            entries.append(entry)
            print(threshold, entry['archived_fold_routed'], entry['model_expected_recoveries'],
                  entry['timing']['speedup'], flush=True)
        _write_json(args.out / 'route-pilot.json', entries)
        return
    if args.phase == 'lock':
        if config.exists():
            raise ValueError('Plan already frozen')
        if args.table_bits is None or args.ordered is None:
            raise ValueError('Choose a pilot-tested configuration before freezing the plan')
        if args.route_cost and (args.ordered or args.table_bits != 21):
            raise ValueError('Routing cutoffs were piloted only with the 21-bit reversal-canonical cache')
        plan = dict(physical=PHYSICAL, reverse=not args.ordered, table_bits=args.table_bits,
                    study_sha256=_sha(Path(__file__)),
                    route_cost=args.route_cost, native_sha256=_sha(native_library_path()) if args.route_cost else None,
                    route_pilot_sha256=_sha(args.out / 'route-pilot.json') if args.route_cost else None,
                    hashes=hashes, seeds=TEST_SEEDS, shots_per_seed=500_000, timing_seed=84598010,
                    timing_shots=TIMING_BATCHES * TIMING_SHOTS + BLOCK, timing_passes=TIMING_BATCHES, batch=BLOCK,
                    pilot_sha256=_sha(args.out / 'pilot.json'),
                    versions={p: version(p) for p in ('stim', 'numpy', 'pymatching', 'scipy')})
        _write_json(config, plan)
        return
    plan = _read_json(config)
    if (hashes != plan['hashes'] or _sha(args.out / 'pilot.json') != plan['pilot_sha256'] or
            _sha(Path(__file__)) != plan['study_sha256']):
        raise ValueError('Archive, pilot files, and study code must agree with the frozen plan')
    if plan['route_cost'] and (_sha(native_library_path()) != plan['native_sha256'] or
                              _sha(args.out / 'route-pilot.json') != plan['route_pilot_sha256']):
        raise ValueError('Native backend and routing pilot must agree with the frozen plan')
    start = perf_counter()
    candidate = CachedTeacher(ref, training, reverse=plan['reverse'], table_bits=plan['table_bits'],
                              route_cost=plan['route_cost'])
    setup = perf_counter() - start
    del training
    rows = []
    if args.phase == 'evaluate':
        if (args.out / 'results.json').exists():
            raise ValueError('Evaluation is complete. Use verify to replay it.')
        timing_input, _ = _sample(ref.circuit, plan['timing_seed'], plan['timing_shots'])
        times = _time(candidate, timing_input, plan['timing_passes'], distinct=True)
    for seed in plan['seeds']:
        packed, truth = _sample(ref.circuit, seed, plan['shots_per_seed'])
        base = _batched(candidate.mwpm, packed)
        batches = [candidate.decode(packed[i:i + BLOCK], diagnostics=True) for i in range(0, len(packed), BLOCK)]
        predicted, hits, routed = (np.concatenate(values) for values in zip(*batches))
        path = args.out / f'paired-{seed}.npz'
        arrays = dict(detectors=packed, truth=truth, mwpm=base, cached_teacher=predicted, hits=hits, routed=routed)
        if args.phase == 'verify':
            with np.load(path, allow_pickle=False) as saved:
                for name, values in arrays.items():
                    np.testing.assert_array_equal(saved[name], values)
        else:
            _save_npz(path, **arrays)
        rows.append(_paired(seed, truth, base, predicted, hits, routed))
        print(rows[-1], flush=True)
    result = _summary(rows)
    if args.phase == 'verify':
        saved = _read_json(args.out / 'results.json')
        assert saved['per_seed'] == rows
        assert saved['accuracy'] == json.loads(json.dumps(result))
        print('Reproduced all paired records and accuracy statistics.')
    else:
        _write_json(args.out / 'results.json', dict(
            accuracy=result, per_seed=rows, timing=times, archive_load_seconds=archive_load_seconds,
            setup_seconds=setup, cache_bytes=candidate.table.nbytes, unique_teacher_keys=candidate.unique,
            retained_teacher_keys=candidate.retained))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('pilot', 'route-pilot', 'lock', 'evaluate', 'verify'))
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--table-bits', type=int, choices=(19, 21))
    parser.add_argument('--ordered', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--route-cost', type=float, choices=ROUTE_COSTS)
    run(parser.parse_args())
