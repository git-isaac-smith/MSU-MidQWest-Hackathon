import os

import numpy as np
import pytest

from bascd.batch_workers import ParallelBatchDecoder, packed_decoder_factory
from bascd.circuits import memory_circuit
from bascd.learned_fault_router import LearnedFaultRouter


def _make_identifying_decoder(offset, *, dtype):
    def predict(records):
        return np.column_stack((
            records[:, 0].astype(dtype) + offset,
            np.full(len(records), os.getpid(), dtype=dtype),
        ))
    return predict


def _fail_to_compile():
    raise ValueError("broken factory")


def test_spawned_workers_preserve_order_dtype_and_worker_local_state():
    records = np.arange(33, dtype=np.uint8).reshape(11, 3)
    pool = ParallelBatchDecoder(_make_identifying_decoder, (17,), workers=2,
                                factory_kwargs={"dtype": np.int64})
    with pool:
        assert pool.startup_seconds > 0
        assert len(set(pool.worker_pids)) == 2
        assert os.getpid() not in pool.worker_pids
        pool.warmup(records)
        assert pool.warmup_seconds > 0
        result = pool.decode_batch(records)
        assert result.dtype == np.int64
        assert pool.decode_batch(records[:0]).shape == (0, 2)
        np.testing.assert_array_equal(result[:, 0], records[:, 0] + 17)
        np.testing.assert_array_equal(result[:, 1], np.repeat(pool.worker_pids, [6, 5]))
        np.testing.assert_array_equal(pool.decode_batch(records[:1]), result[:1])
    assert pool.worker_pids == ()
    with pytest.raises(RuntimeError, match="Open the worker context"):
        pool.decode_batch(records)


def test_worker_compilation_failure_reaches_caller_and_clears_context():
    pool = ParallelBatchDecoder(_fail_to_compile, workers=1)
    with pytest.raises(RuntimeError, match="broken factory"):
        with pool:
            pytest.fail("worker initialization failed before context entry")
    assert pool.worker_pids == ()


@pytest.mark.parametrize("mode", ["ordinary", "correlated", "activity", "learned"])
def test_standard_decoder_parallel_predictions_match_serial(mode, tmp_path):
    circuit = memory_circuit("z", 3, 3, .005)
    text = str(circuit)
    records = circuit.compile_detector_sampler(seed=9042920).sample(127, bit_packed=True)
    kwargs = {}
    if mode == "activity":
        kwargs["calibration_seed"] = 9042921
    if mode == "learned":
        model = circuit.detector_error_model(decompose_errors=True)
        policy = LearnedFaultRouter.fit_sampled(circuit, model, seeds=[9042922], shots_per_seed=1000)
        path = tmp_path / "learned.npz"
        policy.save(path)
        kwargs["policy_path"] = path
    serial = packed_decoder_factory(mode, text, **kwargs)
    with ParallelBatchDecoder(packed_decoder_factory, (mode, text), workers=2, factory_kwargs=kwargs) as pool:
        pool.warmup(records[:32])
        result = pool.decode_batch(records)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, serial(records))
