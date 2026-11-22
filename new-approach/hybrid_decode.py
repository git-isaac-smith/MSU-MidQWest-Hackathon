"""Decode detector-only Stim b8 files with a locked hybrid policy."""

import argparse
from operator import index
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryFile

if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from hybrid_policy import HybridDecoder


MODES = ("balanced", "learned", "accuracy", "margin_only", "full_map", "distance7")
DEFAULT_BATCH_SIZE = 4096


def decode_file(decoder, detectors, out, *, mode="balanced", batch_size=DEFAULT_BATCH_SIZE):
    """Read one detector record and write one logical prediction per shot in Stim b8 format."""
    batch_size = index(batch_size)
    if batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    if mode not in MODES:
        raise ValueError(f"Unknown decoding mode {mode}")
    circuit = decoder.ref.circuit
    detector_count = circuit.num_detectors
    if detector_count < 1 or circuit.num_observables != 1:
        raise ValueError("The policy circuit needs at least one detector and exactly one observable")

    source_path, out_path = Path(detectors), Path(out)
    if os.path.lexists(out_path):
        raise FileExistsError(f"Output already exists: {out_path}")
    row_bytes = (detector_count + 7) // 8
    padding_mask = (0xff << (detector_count % 8)) & 0xff if detector_count % 8 else 0
    total = 0
    with source_path.open("rb") as source, TemporaryFile(mode="w+b") as staged:
        partial_record = f"Partial b8 detector record. The policy expects {row_bytes} bytes per shot"
        if os.fstat(source.fileno()).st_size % row_bytes:
            raise ValueError(partial_record)
        while block := source.read(batch_size * row_bytes):
            if len(block) % row_bytes:
                raise ValueError(partial_record)
            packed = np.frombuffer(block, dtype=np.uint8).reshape(-1, row_bytes)
            if padding_mask and np.any(packed[:, -1] & padding_mask):
                raise ValueError("Nonzero padding bits in a b8 detector record")
            rows = np.unpackbits(packed, axis=1, count=detector_count, bitorder="little").view(np.bool_)
            prediction = np.asarray(decoder.predict(rows, mode=mode))
            if (
                prediction.shape != (len(rows),)
                or prediction.dtype.kind not in "bui"
                or np.any((prediction != 0) & (prediction != 1))
            ):
                raise ValueError("The decoder must return one binary logical prediction per shot")
            staged.write(prediction.astype(np.uint8, copy=False).tobytes())
            total += len(rows)

        staged.seek(0)
        # Keep the destination exclusive even if it appeared while decoding.
        output = out_path.open("xb")
        try:
            with output:
                copyfileobj(staged, output)
        except BaseException:
            out_path.unlink(missing_ok=True)
            raise
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Locked hybrid policy directory")
    parser.add_argument("--detectors", type=Path, required=True, help="Stim b8 detector file in policy circuit order")
    parser.add_argument("--out", type=Path, required=True, help="New b8 file with one observable prediction per shot")
    parser.add_argument("--mode", choices=MODES, default="balanced")
    parser.add_argument("--posterior-backend", choices=("cached", "native", "reference"),
                        help="Full MAP backend (default: portable NumPy cache)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer")
    try:
        if args.posterior_backend is not None and args.mode != "full_map":
            parser.error("--posterior-backend applies only to full_map")
        if args.mode == "full_map":
            from full_map_decoder import FullMapDecoder
            decoder = FullMapDecoder.load(args.policy, args.posterior_backend or "cached")
        elif args.mode == "distance7":
            from distance7_study import DeployedDistanceSeven
            decoder = DeployedDistanceSeven(args.policy)
        else:
            decoder = HybridDecoder.load(args.policy)
        total = decode_file(decoder, args.detectors, args.out, mode=args.mode, batch_size=args.batch_size)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Decoded {total:,} shots in {args.mode} mode to {args.out}")


if __name__ == "__main__":
    main()
