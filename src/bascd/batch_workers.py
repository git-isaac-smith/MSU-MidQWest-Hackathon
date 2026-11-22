"""Synchronous packed-batch decoding with persistent independent workers."""

from functools import partial
import multiprocessing as mp
from time import perf_counter
from traceback import format_exc

import numpy as np


_PACKED = dict(bit_packed_shots=True, bit_packed_predictions=True)


def packed_decoder_factory(mode, circuit_text, *, calibration_seed=None, fraction=.06, policy_path=None):
    """Build a standard packed decoder from a circuit and optional routing policy."""
    if mode not in ("ordinary", "correlated", "activity", "learned"):
        raise ValueError(f"Unknown packed decoder mode {mode!r}")

    import pymatching
    import stim

    circuit = stim.Circuit(circuit_text)
    model = circuit.detector_error_model(decompose_errors=True)
    if mode in ("ordinary", "correlated"):
        correlated = mode == "correlated"
        matching = pymatching.Matching.from_detector_error_model(model, enable_correlations=correlated)
        return partial(matching.decode_batch, enable_correlations=correlated, **_PACKED)
    if mode == "activity":
        from .css_routed import CssRoutedDecoder

        if calibration_seed is None:
            raise ValueError("Activity routing requires a fixed calibration seed")
        return CssRoutedDecoder(circuit, model, calibration_seed=calibration_seed, fraction=fraction).decode_batch

    from .learned_fault_router import LearnedFaultRouter, LearnedFaultRoutedDecoder

    if policy_path is None:
        raise ValueError("Learned routing requires a frozen policy path")
    return LearnedFaultRoutedDecoder(model, LearnedFaultRouter.load(policy_path)).decode_batch


def _run_worker(connection, factory, args, kwargs):
    with connection:
        try:
            instance = factory(*args, **kwargs)
            decode = getattr(instance, "decode_batch", instance)
            if not callable(decode):
                raise TypeError("The worker factory must return a callable or an object with decode_batch")
            connection.send((True, None))
            while (records := connection.recv()) is not None:
                prediction = np.asarray(decode(records))
                if prediction.ndim == 0 or len(prediction) != len(records):
                    raise ValueError("The decoder must return one prediction row per input row")
                connection.send((True, prediction))
        except (EOFError, BrokenPipeError):
            return
        except Exception:
            connection.send((False, format_exc()))


class ParallelBatchDecoder:
    """Use one spawned process and compiled decoder per worker.

    Use a top-level picklable factory and protect executable script entry
    points with the standard multiprocessing main guard. The context starts
    workers and waits for compilation. Call warmup with at least one shot per
    worker before timing. Decode time includes NumPy transfer through pipes,
    decoding, and ordered result assembly. No worker is launched per job.
    Calls on one context are synchronous and should use one caller thread.
    """

    transport = "spawned processes with multiprocessing Pipe and NumPy IPC"

    def __init__(self, factory, args=(), *, workers=2, factory_kwargs=None):
        if type(workers) is not int or workers < 1:
            raise ValueError("Expected a positive integer worker count")
        self.factory = factory
        self.args = tuple(args)
        self.factory_kwargs = dict(factory_kwargs or {})
        self.workers = workers
        self.startup_seconds = None
        self.warmup_seconds = None
        self._workers = []

    @property
    def worker_pids(self):
        """Return the process IDs of the open worker processes."""
        return tuple(process.pid for process, _ in self._workers)

    def __enter__(self):
        if self._workers:
            raise RuntimeError("The worker context is already open")
        context = mp.get_context("spawn")
        start = perf_counter()
        self.warmup_seconds = None
        try:
            for _ in range(self.workers):
                owner, child = context.Pipe()
                try:
                    process = context.Process(
                        target=_run_worker, args=(child, self.factory, self.args, self.factory_kwargs))
                    process.start()
                except BaseException:
                    owner.close()
                    raise
                finally:
                    child.close()
                self._workers.append((process, owner))
            for process, owner in self._workers:
                self._receive(process, owner)
        except BaseException:
            self._shutdown(abort=True)
            raise
        self.startup_seconds = perf_counter() - start
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._shutdown(abort=False)

    @staticmethod
    def _receive(process, connection):
        try:
            ok, value = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(f"Decoder worker {process.pid} exited before returning a result") from exc
        if not ok:
            raise RuntimeError(f"Decoder worker {process.pid} failed:\n{value}")
        return value

    def _shutdown(self, *, abort):
        workers, self._workers = self._workers, []
        for process, owner in workers:
            if abort:
                if process.is_alive():
                    process.terminate()
            else:
                try:
                    owner.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
            owner.close()
        for process, _ in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()

    def warmup(self, records):
        """Run one warm batch on all workers and record its separate duration."""
        if len(records) < self.workers:
            raise ValueError("Warmup requires at least one shot per worker")
        start = perf_counter()
        self.decode_batch(records)
        self.warmup_seconds = perf_counter() - start

    def decode_batch(self, records):
        """Decode contiguous chunks and return predictions in input row order."""
        if not self._workers:
            raise RuntimeError("Open the worker context before decoding")
        if not isinstance(records, np.ndarray) or records.ndim != 2 or records.dtype != np.uint8:
            raise ValueError("Expected a two-dimensional packed uint8 detector batch")
        active = self._workers[:max(1, min(self.workers, len(records)))]
        chunks = np.array_split(records, len(active), axis=0)
        try:
            for (_, owner), chunk in zip(active, chunks):
                owner.send(chunk)
            predictions = [self._receive(process, owner) for process, owner in active]
        except BaseException:
            self._shutdown(abort=True)
            raise
        first = predictions[0]
        if any(part.shape[1:] != first.shape[1:] or part.dtype != first.dtype for part in predictions[1:]):
            raise ValueError("Decoder workers returned inconsistent prediction shapes or dtypes")
        return np.concatenate(predictions, axis=0) if len(predictions) > 1 else first
