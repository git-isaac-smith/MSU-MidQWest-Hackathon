"""Time within-batch syndrome deduplication and compare every output bit."""

import argparse
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from bascd.accuracy_campaign import draw
from bascd.accuracy_decoders import load_frozen_decoder
from bascd.records import manifest, save_json, scenarios
from bascd.unique_batch import UniqueBatchDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=941150)
    args = parser.parse_args()
    if args.shots < 2 or args.out.exists():
        parser.error("choose a new output directory and at least two shots")
    cases = scenarios((5,7), (.001,.01), ("z",))
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = manifest(["python", *sys.argv], args.seed, args.shots, 3)
    protocol.update(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                    split="validation", cases="d=5,7, Z memory, p=.001,.01",
                    wrapper=dict(chunk_size=32768, probe_shots=1024, min_duplicate_fraction=.01))
    save_json(args.out / "manifest.json", protocol)
    rows = []
    for i, case in enumerate(cases):
        circuit, _, data = draw(case, args.seed, "validation", args.shots)
        decoder = load_frozen_decoder(args.policy, circuit)
        wrapped = UniqueBatchDecoder(decoder, circuit.num_detectors, **protocol["wrapper"])
        paths = {"direct":decoder, "deduplicated":wrapped}
        times = {name:[] for name in paths}
        for model in paths.values():
            model.decode_batch(data.detectors[:32])
        predictions = {}
        for repeat in range(3):
            names = list(paths) if (i+repeat)%2 == 0 else list(reversed(paths))
            for name in names:
                start = perf_counter()
                pred = paths[name].decode_batch(data.detectors)
                times[name].append(perf_counter()-start)
                predictions[name] = pred
            if not np.array_equal(predictions["direct"], predictions["deduplicated"]):
                raise ValueError("within-batch deduplication changed logical predictions")
        packed = np.packbits(data.detectors, axis=1, bitorder="little")
        rows.append(dict(**data.metadata, unique_syndromes=len(np.unique(packed,axis=0)),
                         prediction_sha256=hashlib.sha256(predictions["direct"].tobytes()).hexdigest(),
                         batch_times_s=times, mismatching_predictions=0,
                         median_us_per_shot={k:1e6*float(np.median(v))/args.shots for k,v in times.items()}))
        save_json(args.out / "measurements.json", rows)
        print(case, rows[-1]["median_us_per_shot"], flush=True)


if __name__ == "__main__":
    main()
