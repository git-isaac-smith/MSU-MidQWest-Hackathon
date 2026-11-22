"""Exhaust every syndrome and herald mask on a small tied, re-scalable graph."""
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pymatching
from engine import FastErasureDecoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sakidja-table5-replay-01"))
from replay import load_upstream


def main():
    author_class = load_upstream(Path(sys.argv[1])).ErasureDecoder

    class Fixture(author_class):
        def __init__(self, _, mask):
            self.is_herald = mask
            self.skipped = 0
            # The first herald removes the unique maximum, the next removes the
            # remaining fractional weights. The last erases every internal edge.
            self.edges = {
                (0, 1): (7.123456789, set()), (0, 2): (1.234567, {0}),
                (1, 3): (1.234567, set()), (2, 3): (1., set()),
                (0,): (2., {0}), (3,): (2., set()),
            }
            self.cond = {0: [(0, 1)], 1: [(0, 2), (1, 3)], 2: [(0,), (3,)],
                         3: [(0, 1), (0, 2), (1, 3), (2, 3)]}
            self.m = pymatching.Matching()
            for key, (weight, obs) in self.edges.items():
                self._set(key, weight, obs, "disallow")

    mask = np.array([True, False] * 4)
    records = np.array(list(product((False, True), repeat=8)))
    expected = Fixture(None, mask).decode(records)
    scorer = FastErasureDecoder(SimpleNamespace(ErasureDecoder=Fixture), None, mask)
    got = scorer.decode(records)
    np.testing.assert_array_equal(expected, got)
    np.testing.assert_array_equal(expected[::-1], scorer.decode(records[::-1]))
    print(json.dumps(dict(combinations=len(got), positive_predictions=int(expected.sum()), mismatches=0)))


if __name__ == "__main__":
    main()
