r"""Optional C++ backend for the fixed five-round posterior decoder.

Create a ``build`` directory in the repository root. In a Visual Studio x64
developer prompt, build with::

    cl /nologo /O2 /std:c++17 /EHsc /LD /arch:AVX2 new-approach\native_posterior.cpp /Fobuild\native_posterior.obj /link /OUT:build\native_posterior.dll /IMPLIB:build\native_posterior.lib

On Linux build with::

    c++ -O3 -std=c++17 -shared -fPIC new-approach/native_posterior.cpp -o build/libnative_posterior.so

On macOS build with::

    c++ -O3 -std=c++17 -dynamiclib new-approach/native_posterior.cpp -o build/libnative_posterior.dylib

Omit ``/arch:AVX2`` on Windows without AVX2. Add ``-mavx2`` to the ``c++``
commands on AVX2 systems. Pass ``library_path`` to load from another location.
"""

import ctypes
import sys
from functools import lru_cache
from operator import index
from pathlib import Path
from threading import Lock
from weakref import finalize

import numpy as np

from posterior_decoder import PosteriorDecoder
from stim_data import ROUNDS


def native_library_path() -> Path:
    """Return the default path of the explicitly built shared library."""
    if sys.platform == "win32":
        name = "native_posterior.dll"
    elif sys.platform == "darwin":
        name = "libnative_posterior.dylib"
    else:
        name = "libnative_posterior.so"
    return Path(__file__).resolve().parents[1] / "build" / name


@lru_cache(maxsize=4)
def _load_library(path: Path) -> ctypes.CDLL:
    import numpy.ctypeslib as npct

    library = ctypes.CDLL(str(path))
    doubles = npct.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    words = npct.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
    library.posterior_create.argtypes = [doubles, doubles, words, doubles, ctypes.c_size_t, ctypes.c_size_t]
    library.posterior_create.restype = ctypes.c_void_p
    library.posterior_destroy.argtypes = [ctypes.c_void_p]
    library.posterior_destroy.restype = None
    library.posterior_predict.argtypes = [ctypes.c_void_p, words, words, ctypes.c_size_t, doubles]
    library.posterior_predict.restype = ctypes.c_int
    return library


class NativePosteriorDecoder(PosteriorDecoder):
    """Evaluate both logical classes using nonnegative masses in native code.

    Distances other than five and extreme rates use the reference implementation.
    The optional ``prefix_cache_size`` sets the number of retained two-check
    messages, at 64 KiB per message. Up to 4,096 one-check messages are also
    cached on demand.
    """

    def __init__(self, distance: int, p_data: float, p_readout: float, *,
                 library_path: str | Path | None = None, prefix_cache_size: int = 2048):
        super().__init__(distance, p_data, p_readout)
        self._native = None
        if self._check_count != 12 or ROUNDS != 5 or self._use_log:
            return

        prefix_cache_size = index(prefix_cache_size)
        if not 0 < prefix_cache_size <= self._words ** 2:
            raise ValueError("prefix_cache_size must be between one and the number of two-check messages")
        path = Path(library_path).resolve() if library_path is not None else native_library_path()
        library = _load_library(path)
        shifts = np.array([permutation[0] for permutation, _ in self._faults], dtype=np.uint16)
        probabilities = np.array([probability for _, probability in self._faults], dtype=np.float64)
        readout = np.asarray(self._readout, dtype=np.float64)
        handle = library.posterior_create(self._prior, readout, shifts, probabilities,
                                          len(shifts), prefix_cache_size)
        if not handle:
            raise MemoryError("could not initialize native posterior decoder")
        self._native = library.posterior_predict
        self._native_handle = handle
        self._native_lock = Lock()
        finalize(self, library.posterior_destroy, handle)

    def _predict_batch(self, observed: np.ndarray, terminal: np.ndarray) -> np.ndarray:
        if self._native is None:
            return super()._predict_batch(observed, terminal)
        observed = np.ascontiguousarray(observed, dtype=np.uint16)
        terminal = np.ascontiguousarray(terminal, dtype=np.uint16)
        posterior = np.empty(len(observed), dtype=np.float64)
        with self._native_lock:
            status = self._native(self._native_handle, observed, terminal, len(observed), posterior)
        if status == 1:
            raise ValueError("check history has zero probability under the configured noise")
        if status != 0:
            raise MemoryError("native posterior decoder could not allocate a boundary message")
        return posterior
