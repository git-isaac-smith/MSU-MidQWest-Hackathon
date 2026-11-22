"""Dense-shot adapter for the pinned single-logical erasure-matching experiment."""
import ctypes
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

LIBRARY = Path(__file__).resolve().parent / "build" / ("bath_engine.dll" if sys.platform == "win32" else "bath_engine.so")


class FastErasureDecoder:
    def __init__(self, upstream, prior, is_herald):
        if version("pymatching") != "2.4.0":
            raise ValueError("This scorer implements PyMatching 2.4.0")
        reference = upstream.ErasureDecoder(prior, is_herald)
        self.is_herald = np.asarray(is_herald, dtype=bool)
        self.skipped = reference.skipped
        self.nodes = reference.m.num_detectors
        if self.skipped or reference.m.num_fault_ids != 1:
            raise ValueError("Expected a graphlike prior with exactly one logical observable")
        keys = list(reference.edges)
        edge_id = {key: idx for idx, key in enumerate(keys)}
        self.ends = np.array([(k[0], k[1] if len(k) == 2 else -1) for k in keys], dtype=np.int32)
        self.weights = np.array([reference.edges[k][0] for k in keys], dtype=np.float64)
        if (not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0)
                or np.any(self.weights > 2**24 - 1) or len(keys) != reference.m.num_edges
                or np.any(self.ends[:, 0] == self.ends[:, 1])):
            raise ValueError("Unsupported matching graph edge")
        self.observables = np.array([bool(reference.edges[k][1]) for k in keys], dtype=np.uint8)
        self.heralds = int(self.is_herald.sum())
        offsets = [0]
        conditioned = []
        for h in range(self.heralds):
            conditioned.extend(edge_id[k] for k in reference.cond.get(h, ()))
            offsets.append(len(conditioned))
        self.offsets = np.asarray(offsets, dtype=np.int32)
        self.conditioned = np.asarray(conditioned, dtype=np.int32)
        self._lib = ctypes.CDLL(str(LIBRARY))
        self._score = self._lib.score
        i = ctypes.c_int
        u8 = ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        i32 = ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        f64 = ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self._score.argtypes = [i, i, i32, f64, u8, i, i32, i32, i, u8, u8, u8]
        self._score.restype = ctypes.c_char_p

    def decode(self, detectors):
        detectors = np.asarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != len(self.is_herald) or len(detectors) > 2**31-1:
            raise ValueError("Expected an unpacked shot-by-detector matrix")
        herald = np.ascontiguousarray(detectors[:, self.is_herald], dtype=np.uint8)
        syndrome = np.ascontiguousarray(detectors[:, ~self.is_herald][:, :self.nodes], dtype=np.uint8)
        predictions = np.empty((len(detectors), 1), dtype=np.uint8)
        error = self._score(self.nodes, len(self.weights), self.ends, self.weights, self.observables,
                            self.heralds, self.offsets, self.conditioned, len(detectors),
                            herald, syndrome, predictions)
        if error:
            raise ValueError(error.decode())
        return predictions
