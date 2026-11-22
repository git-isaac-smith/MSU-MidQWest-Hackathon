"""Portable posterior decisions for packed detector records with one observable."""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pymatching
import stim
from scipy.stats import binomtest

from .exact_small import MAX_DETECTORS, ExactSmallDecoder, compile_exact_small
from .stats import wilson95


FORMAT = "bascd-packed-posterior-v1"
SCORE_SEED = 89503101
WARMUP_SEED = 89503102
SINGLE_SEED = 89503103
BATCH_SEED = 89503104


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metadata_digest(metadata: dict) -> str:
    payload = {key: value for key, value in metadata.items() if key != "manifest_sha256"}
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


@dataclass(frozen=True)
class PortableLookup:
    """Index a little-endian packed decision bit by the raw detector integer."""

    num_detectors: int
    table: bytes
    _width: int = field(init=False, repr=False)
    _limit: int = field(init=False, repr=False)
    _array: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.num_detectors) is not int or not 1 <= self.num_detectors <= MAX_DETECTORS:
            raise ValueError(f"expected 1 to {MAX_DETECTORS} detector bits")
        entries = 1 << self.num_detectors
        if not isinstance(self.table, bytes) or len(self.table) != (entries + 7) // 8:
            raise ValueError("table byte count does not match the detector width")
        object.__setattr__(self, "_width", (self.num_detectors + 7) // 8)
        object.__setattr__(self, "_limit", entries)
        object.__setattr__(self, "_array", np.frombuffer(self.table, dtype=np.uint8))

    @property
    def record_bytes(self) -> int:
        return self._width

    @classmethod
    def from_exact(cls, decoder: ExactSmallDecoder):
        """Materialize logical decisions in the sampler's raw detector frame."""
        count = 1 << decoder.num_detectors
        if decoder._detector_offset == 0 and not decoder._observable_offset:
            return cls(decoder.num_detectors, np.packbits(decoder._negative, bitorder="little").tobytes())
        packed = bytearray()
        for start in range(0, count, 1 << 16):
            indices = np.arange(start, min(start + (1 << 16), count), dtype=np.uint32)
            predictions = decoder._negative[indices ^ decoder._detector_offset] ^ decoder._observable_offset
            packed.extend(np.packbits(predictions, bitorder="little").tobytes())
        return cls(decoder.num_detectors, bytes(packed))

    def predict(self, packed_record: bytes) -> int:
        """Return a one-bit integer prediction for a complete detector record."""
        if not isinstance(packed_record, bytes) or len(packed_record) != self._width:
            raise ValueError(f"expected a bytes record of length {self._width}")
        index = int.from_bytes(packed_record, "little")
        if index >= self._limit:
            raise ValueError("detector padding bits must be zero")
        return (self.table[index >> 3] >> (index & 7)) & 1

    def predict_batch(self, packed_rows: np.ndarray) -> np.ndarray:
        """Return one packed uint8 prediction column for uint8 detector rows."""
        rows = np.asarray(packed_rows)
        if rows.ndim != 2 or rows.shape[1] != self._width or rows.dtype != np.uint8:
            raise ValueError(f"expected uint8 rows with {self._width} detector bytes")
        if self.num_detectors % 8 and np.any(rows[:, -1] >> (self.num_detectors % 8)):
            raise ValueError("detector padding bits must be zero")
        indices = rows[:, 0].astype(np.uint32)
        for column in range(1, self._width):
            indices |= rows[:, column].astype(np.uint32) << (8 * column)
        decoded = (self._array[indices >> 3] >> (indices & 7)) & 1
        return decoded.astype(np.uint8)[:, None]

    @classmethod
    def load(cls, directory: Path):
        """Load an export after checking the manifest, circuit and table hashes."""
        directory = Path(directory)
        metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if metadata.get("manifest_sha256") != _metadata_digest(metadata):
            raise ValueError("lookup manifest SHA-256 mismatch")
        bits = metadata.get("detector_bits")
        if (metadata.get("format") != FORMAT or type(bits) is not int or not 1 <= bits <= MAX_DETECTORS
                or metadata.get("record_bytes") != (bits + 7) // 8 or metadata.get("observable_bits") != 1
                or metadata.get("index_byte_order") != "little" or metadata.get("decision_bit_order") != "little"):
            raise ValueError("unsupported lookup format")
        table = (directory / "decisions.b8").read_bytes()
        circuit = (directory / "circuit.stim").read_bytes()
        if _sha256(table) != metadata.get("table_sha256") or len(table) != metadata.get("table_bytes"):
            raise ValueError("lookup table SHA-256 or length mismatch")
        if _sha256(circuit) != metadata.get("circuit_sha256"):
            raise ValueError("lookup circuit SHA-256 mismatch")
        return cls(bits, table)


def _write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _compile(circuit_path: Path):
    source = circuit_path.read_bytes()
    circuit = stim.Circuit(source.decode("utf-8"))
    start = perf_counter_ns()
    exact = compile_exact_small(circuit.detector_error_model())
    exact_s = (perf_counter_ns() - start) / 1e9
    start = perf_counter_ns()
    lookup = PortableLookup.from_exact(exact)
    pack_s = (perf_counter_ns() - start) / 1e9
    return source, circuit, exact, lookup, exact_s, pack_s


def _export(directory: Path, source: bytes, exact: ExactSmallDecoder, lookup: PortableLookup):
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "format": FORMAT,
        "detector_bits": lookup.num_detectors,
        "record_bytes": lookup.record_bytes,
        "observable_bits": 1,
        "index_byte_order": "little",
        "decision_bit_order": "little",
        "table_bytes": len(lookup.table),
        "table_sha256": _sha256(lookup.table),
        "circuit_sha256": _sha256(source),
        "model": "Undecomposed Stim detector error model",
        "estimated_optimal_ler": exact.estimated_optimal_ler,
        "ml_excess_ler_bound": exact.excess_ler_bound,
        "uncertain_syndromes": exact.uncertain_syndromes,
    }
    metadata["manifest_sha256"] = _metadata_digest(metadata)
    (directory / "circuit.stim").write_bytes(source)
    (directory / "decisions.b8").write_bytes(lookup.table)
    _write_json(directory / "manifest.json", metadata)


def predict_file(lookup: PortableLookup, source: Path, output: Path) -> None:
    """Decode a b8 detector file to one observable byte per record."""
    with source.open("rb") as incoming:
        if os.fstat(incoming.fileno()).st_size % lookup.record_bytes:
            raise ValueError("input contains a partial detector record")
        with output.open("xb") as outgoing:
            try:
                while block := incoming.read(4096 * lookup.record_bytes):
                    records = np.frombuffer(block, dtype=np.uint8).reshape(-1, lookup.record_bytes)
                    outgoing.write(lookup.predict_batch(records).tobytes())
            except (ValueError, OSError):
                outgoing.close()
                output.unlink()
                raise


def _scalar_wrappers(lookup, matching):
    def packed_mwpm(record):
        rows = np.frombuffer(record, dtype=np.uint8).reshape(1, 3)
        return int(matching.decode_batch(rows, bit_packed_shots=True, bit_packed_predictions=True)[0, 0])

    def unpacked_mwpm(record):
        bits = np.unpackbits(np.frombuffer(record, dtype=np.uint8), bitorder="little")
        return int(matching.decode(bits)[0])

    return {"portable_lookup": lookup.predict, "mwpm_packed_one_row": packed_mwpm, "mwpm_unpacked_scalar": unpacked_mwpm}


def _timing_statistics(times):
    values = np.asarray(times)
    return {key: float(np.percentile(values, quantile)) for key, quantile in (
        ("median", 50), ("p95", 95), ("p99", 99),
    )}


def _benchmark(source_path: Path, directory: Path, shots: int) -> None:
    if shots < 100_000:
        raise ValueError("the benchmark requires at least 100,000 scoring shots")
    if directory.exists():
        raise FileExistsError(directory)
    source, circuit, exact, lookup, exact_s, pack_s = _compile(source_path)
    if circuit.num_detectors != 24:
        raise ValueError("the timing study requires a 24-detector circuit")
    start = perf_counter_ns()
    matching = pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))
    mwpm_s = (perf_counter_ns() - start) / 1e9
    score_rows, truth = circuit.compile_detector_sampler(seed=SCORE_SEED).sample(
        shots, bit_packed=True, separate_observables=True
    )
    reference = exact.decode_batch(np.unpackbits(score_rows, axis=1, bitorder="little"))
    predictions = lookup.predict_batch(score_rows)
    disagreements = int(np.count_nonzero(predictions != reference))
    if disagreements:
        raise ValueError(f"portable lookup disagreed with exact-small on {disagreements} shots")
    scalar = _scalar_wrappers(lookup, matching)
    batch = {
        "portable_lookup": lookup.predict_batch,
        "mwpm_packed_batch": lambda rows: matching.decode_batch(
            rows, bit_packed_shots=True, bit_packed_predictions=True
        ),
    }
    failed = predictions != truth
    baseline_failed = batch["mwpm_packed_batch"](score_rows) != truth
    fixed = int((baseline_failed & ~failed).sum())
    added = int((failed & ~baseline_failed).sum())

    warm = circuit.compile_detector_sampler(seed=WARMUP_SEED).sample(1024, bit_packed=True)
    for method in scalar.values():
        for row in warm:
            method(row.tobytes())
    for method in batch.values():
        method(warm)

    scalar_rows = circuit.compile_detector_sampler(seed=SINGLE_SEED).sample(20000, bit_packed=True)
    expected_scalar = batch["portable_lookup"](scalar_rows)[:, 0]
    expected_mwpm = batch["mwpm_packed_batch"](scalar_rows)[:, 0]
    scalar_times = {name: [] for name in scalar}
    methods = list(scalar.items())
    for index, row in enumerate(scalar_rows):
        record = row.tobytes()
        first = index % len(methods)
        for name, method in methods[first:] + methods[:first]:
            start = perf_counter_ns()
            result = method(record)
            elapsed = perf_counter_ns() - start
            expected = expected_scalar if name == "portable_lookup" else expected_mwpm
            if result != expected[index]:
                raise ValueError(f"{name} scalar and batch predictions disagree")
            scalar_times[name].append(elapsed / 1000)

    sampler = circuit.compile_detector_sampler(seed=BATCH_SEED)
    batch_rows = [sampler.sample(8192, bit_packed=True) for _ in range(11)]
    batch_times = {name: [] for name in batch}
    methods = list(batch.items())
    for index, rows in enumerate(batch_rows):
        ordered = methods if index % 2 == 0 else reversed(methods)
        for name, method in ordered:
            start = perf_counter_ns()
            method(rows)
            batch_times[name].append((perf_counter_ns() - start) / (1000 * len(rows)))

    scalar_stats = {name: _timing_statistics(times) for name, times in scalar_times.items()}
    best_mwpm = min((name for name in scalar_stats if name.startswith("mwpm")), key=lambda name: scalar_stats[name]["median"])
    batch_stats = {name: _timing_statistics(times) for name, times in batch_times.items()}
    _export(directory, source, exact, lookup)
    _write_json(directory / "benchmark.json", {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("numpy", "stim", "pymatching", "scipy")},
        },
        "protocol": {
            "score_seed": SCORE_SEED,
            "warmup_seed": WARMUP_SEED,
            "single_seed": SINGLE_SEED,
            "batch_seed": BATCH_SEED,
            "scalar_records": len(scalar_rows),
            "warmup_records": len(warm),
            "batch_repeats": len(batch_rows),
            "batch_records": len(batch_rows[0]),
            "scalar_interface": "three input bytes to one integer bit; conversion included",
            "batch_interface": "packed uint8 detector rows to one uint8 prediction column",
            "timing_scope": "CPU Python call timings, including timer overhead and excluding shot sampling",
        },
        "setup_seconds": {"exact_dem_and_compile": exact_s, "pack_table": pack_s, "mwpm_dem_and_compile": mwpm_s},
        "score": {
            "shots": shots,
            "exact_reference_disagreements": disagreements,
            "lookup_errors": int(failed.sum()),
            "mwpm_errors": int(baseline_failed.sum()),
            "lookup_wilson95": wilson95(int(failed.sum()), shots),
            "mwpm_wilson95": wilson95(int(baseline_failed.sum()), shots),
            "fixed": fixed,
            "added": added,
            "paired_exact_p": float(binomtest(fixed, fixed + added).pvalue) if fixed + added else 1.0,
        },
        "scalar_microseconds_per_shot": scalar_stats,
        "best_mwpm_scalar": best_mwpm,
        "scalar_median_speedup": scalar_stats[best_mwpm]["median"] / scalar_stats["portable_lookup"]["median"],
        "batch_microseconds_per_shot": batch_stats,
        "batch_median_speedup": batch_stats["mwpm_packed_batch"]["median"] / batch_stats["portable_lookup"]["median"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "benchmark"):
        command = commands.add_parser(name)
        command.add_argument("--circuit", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        if name == "benchmark":
            command.add_argument("--shots", type=int, default=150_000)
    predict = commands.add_parser("predict")
    predict.add_argument("--table", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        if args.out.exists():
            raise FileExistsError(args.out)
        source, _, exact, lookup, _, _ = _compile(args.circuit)
        _export(args.out, source, exact, lookup)
    elif args.command == "benchmark":
        _benchmark(args.circuit, args.out, args.shots)
    else:
        predict_file(PortableLookup.load(args.table), args.input, args.out)


if __name__ == "__main__":
    main()
