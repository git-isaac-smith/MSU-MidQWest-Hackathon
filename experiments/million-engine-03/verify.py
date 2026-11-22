"""Compare every prediction with the author on fresh physical and altered herald records."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from engine import FastErasureDecoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bath-background-stress-01"))
from run import build_channels, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--physical", type=int, default=1024)
    args = parser.parse_args()
    upstream = replay.load_upstream(args.upstream)
    rows = []
    for a, name in enumerate(replay.ARMS):
        arm = build_channels(args.upstream, upstream, name)
        for p, background in enumerate((.001, .01, .04)):
            layout = (a + 2 * p) % 5
            draw = np.random.default_rng(upstream.SEED + 7000 + layout).integers(0, len(arm["channels"]), 49)
            upstream.P_CHECK = background
            def build(field):
                return upstream.build_circuit(7, "z", *upstream._rates([arm[field][i] for i in draw]), upstream.P_MEAS)
            physical, mask = build("channels")
            prior, _ = build("priors")
            author = upstream.ErasureDecoder(prior, mask)
            fast = FastErasureDecoder(upstream, prior, mask)
            seed = 745038923100 + 10*a + p
            dets, obs = physical.compile_detector_sampler(seed=seed).sample(args.physical, separate_observables=True)
            rng = np.random.default_rng(seed + 100)
            hard = rng.random((128, len(mask))) < .5
            hard[:32, mask] = True
            hard[32:64, mask] = False
            shots = np.concatenate((dets, hard))
            start = perf_counter()
            expected = author.decode(shots)
            reference_s = perf_counter() - start
            start = perf_counter()
            actual = fast.decode(shots)
            fast_s = perf_counter() - start
            assert np.array_equal(expected, actual), np.flatnonzero(expected[:, 0] != actual[:, 0])
            assert np.array_equal(actual[::-1], fast.decode(shots[::-1]))
            row = dict(arm=name, layout=layout+1, p_check=background, seed=seed,
                       physical_shots=len(dets), altered_shots=len(hard), mismatches=0,
                       physical_failures=int(np.count_nonzero(expected[:len(dets)] != obs)),
                       author_s=reference_s, accelerated_s=fast_s)
            print(row, flush=True)
            rows.append(row)
    (HERE / "validation.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
