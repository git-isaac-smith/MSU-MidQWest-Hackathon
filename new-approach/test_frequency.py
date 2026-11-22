import numpy as np
import pytest

from frequency import Passband, example_spectra, features, retained_energy


def test_all_pass_reconstructs_frames_and_preserves_weighted_stack_and_geometry():
    rng = np.random.default_rng(41)
    frames = rng.normal(size=(131, 5, 3, 5)).astype(np.float32)
    check_mask = np.indices((3, 5)).sum(axis=0) % 2 == 0
    all_pass = Passband(temporal=0.5, spatial=np.sqrt(0.5))

    filtered = features(frames, check_mask, all_pass)
    unfiltered = features(frames, check_mask)

    assert all_pass.mask(3, 5).all()
    assert filtered.shape == (131, 7, 3, 5)
    assert filtered.dtype == np.float32
    np.testing.assert_allclose(filtered, unfiltered, atol=1e-6)
    np.testing.assert_allclose(filtered[:, :5], frames, atol=1e-6)
    np.testing.assert_allclose(
        filtered[:, 5], np.average(frames, axis=1, weights=np.arange(1, 6)), atol=1e-6
    )
    np.testing.assert_array_equal(filtered[:, 6], np.broadcast_to(check_mask, (131, 3, 5)))


def test_temporal_dc_retains_static_frames_and_removes_zero_mean_variation():
    pattern = np.array([[1, 0, -2], [0, 2, 3]], dtype=np.float32)
    static = np.broadcast_to(pattern, (5, 2, 3))
    variation = np.arange(-2, 3, dtype=np.float32)[:, None, None] * pattern
    frames = np.stack([static, variation, static + variation])
    dc = Passband(temporal=0.0, spatial=np.sqrt(0.5))

    filtered = features(frames, np.ones(pattern.shape, dtype=bool), dc)

    np.testing.assert_allclose(
        filtered[[0, 2], :6], np.broadcast_to(pattern, (2, 6, 2, 3)), atol=1e-6
    )
    np.testing.assert_allclose(filtered[1, :6], 0.0, atol=1e-6)


def test_example_spectra_match_complex_fft_and_filtered_feature_stack():
    rng = np.random.default_rng(42)
    shot = rng.normal(size=(5, 3, 4)).astype(np.float32)
    passband = Passband(temporal=0.1, spatial=0.4)
    spatial, stack, filtered_stack, joint, mask = example_spectra(shot, passband)

    reference_spatial = np.fft.fft2(shot, s=(6, 8), axes=(-2, -1))
    reference_joint = np.fft.fft(np.concatenate([reference_spatial, reference_spatial[::-1]]), axis=0)
    reference_filtered = np.fft.ifft(reference_joint * mask, axis=0)[:5]
    assert mask.shape == joint.shape == (10, 6, 8)
    assert mask.any() and not mask.all()
    assert np.any(spatial.imag) and np.any(joint.imag)
    np.testing.assert_allclose(spatial, reference_spatial, atol=1e-5)
    np.testing.assert_allclose(joint, reference_joint, atol=1e-5)
    np.testing.assert_allclose(
        stack, np.average(reference_spatial, axis=0, weights=np.arange(1, 6)), atol=1e-5
    )
    np.testing.assert_allclose(
        filtered_stack, np.average(reference_filtered, axis=0, weights=np.arange(1, 6)), atol=1e-5
    )
    filtered = features(shot[None], np.ones((3, 4), dtype=bool), passband)
    np.testing.assert_allclose(
        np.fft.ifft2(filtered_stack).real[:3, :4], filtered[0, 5], atol=1e-6
    )


def test_retained_energy_pools_fourier_power_and_handles_zero_signal():
    pattern = np.array([[1, 0, -2], [0, 2, 3]], dtype=np.float32)
    dc = Passband(temporal=0.0, spatial=np.sqrt(0.5))
    all_pass = Passband(temporal=0.5, spatial=np.sqrt(0.5))
    frames = np.zeros((130, 5, 2, 3), dtype=np.float32)
    frames[0] = pattern
    frames[-1] = np.arange(-2, 3, dtype=np.float32)[:, None, None] * pattern

    assert retained_energy(frames, dc) == pytest.approx(1 / 3)
    assert retained_energy(frames, all_pass) == pytest.approx(1.0)
    assert retained_energy(frames[1:-1], dc) is None
    assert retained_energy(frames[:0], dc) is None
