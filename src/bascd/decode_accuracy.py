"""Apply a frozen accuracy policy to a Stim b8 file of detector events."""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import stim

from .accuracy_decoders import load_frozen_decoder
from .unique_batch import UniqueBatchDecoder


def decode_file(policy, circuit_path, events_path, output, batch_shots=4096, *, deduplicate=False):
    circuit = stim.Circuit.from_file(circuit_path)
    width = (circuit.num_detectors + 7) // 8
    if width == 0 or batch_shots < 1:
        raise ValueError("a positive batch size and at least one detector are required")
    if output.exists():
        raise FileExistsError(output)
    decoder = load_frozen_decoder(policy, circuit)
    if deduplicate:
        decoder = UniqueBatchDecoder(decoder, circuit.num_detectors)
    count = 0
    # Stage predictions until the complete input has decoded successfully.
    with events_path.open("rb") as src, tempfile.SpooledTemporaryFile(max_size=2**20) as dest:
        while chunk := src.read(batch_shots * width):
            if len(chunk) % width:
                raise ValueError("incomplete final b8 detector record")
            packed = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, width)
            events = np.unpackbits(packed, axis=1, count=circuit.num_detectors, bitorder="little")
            predictions = decoder.decode_batch(events)
            if predictions.shape != (len(events), circuit.num_observables):
                raise ValueError("decoder output shape does not match circuit observables")
            dest.write(np.packbits(predictions, axis=1, bitorder="little").tobytes())
            count += len(events)
        dest.seek(0)
        with output.open("xb") as out:
            shutil.copyfileobj(dest, out)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--circuit", type=Path, required=True)
    parser.add_argument("--dets", type=Path, required=True, help="detector-only Stim b8 file")
    parser.add_argument("--out", type=Path, required=True, help="new observable-prediction b8 file")
    parser.add_argument("--batch-shots", type=int, default=4096)
    parser.add_argument("--deduplicate", action="store_true",
                        help="reuse repeated syndrome predictions within a batch")
    args = parser.parse_args()
    try:
        count = decode_file(args.policy, args.circuit, args.dets, args.out, args.batch_shots,
                            deduplicate=args.deduplicate)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Wrote {count} logical predictions to {args.out}")


if __name__ == "__main__":
    main()
