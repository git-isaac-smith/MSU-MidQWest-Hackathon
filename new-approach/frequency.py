"""Phase-preserving space/time FFT of five observed surface-code check grids."""

from dataclasses import dataclass

import numpy as np
from scipy import fft


WEIGHTS = np.arange(1, 6, dtype=np.float32) / 15


@dataclass(frozen=True)
class Passband:
    temporal: float = 0.20  # cycles per round
    spatial: float = 0.48  # cycles per grid cell, radial

    def __post_init__(self):
        if not 0 <= self.temporal <= 0.5 or not 0 <= self.spatial <= np.sqrt(0.5):
            raise ValueError("Cutoffs must be in [0, 0.5] temporally and [0, sqrt(0.5)] spatially")

    def mask(self, height, width):
        fy = np.fft.fftfreq(2 * height)
        fx = np.fft.fftfreq(2 * width)
        radius = np.hypot(fy[:, None], fx[None, :])
        f = np.abs(np.fft.fftfreq(10))
        return (f[:, None, None] <= self.temporal + 1e-12) & (radius <= self.spatial + 1e-12)


def _spectra(frames):
    height, width = frames.shape[-2:]
    spatial = fft.fft2(frames, s=(2 * height, 2 * width), axes=(-2, -1))
    reflected = np.concatenate([spatial, spatial[:, ::-1]], axis=1)
    return spatial, fft.fft(reflected, axis=1)


def _five_fields(observed):
    observed = np.asarray(observed)
    if observed.ndim != 4 or observed.shape[1] != 5 or min(observed.shape[-2:]) < 1:
        raise ValueError("Expected five check grids with shape (shots, 5, height, width)")
    return observed


def features(observed, check_mask, passband=None):
    """Create seven CNN channels from public measurements and known geometry."""
    observed = _five_fields(observed)
    shots, _, height, width = observed.shape
    if np.shape(check_mask) != (height, width):
        raise ValueError("Check mask must match the spatial grid")
    result = np.empty((shots, 7, height, width), dtype=np.float32)
    mask = passband.mask(height, width) if passband is not None else None
    for start in range(0, shots, 128):
        fields = observed[start : start + 128].astype(np.float32, copy=False)
        if mask is not None:
            _, spectrum = _spectra(fields)
            spatial = fft.ifft(spectrum * mask, axis=1)[:, :5]
            fields = fft.ifft2(spatial, axes=(-2, -1)).real[..., :height, :width]
        block = result[start : start + len(fields)]
        block[:, :5] = fields
        block[:, 5] = np.einsum("t,ntyx->nyx", WEIGHTS, fields)
        block[:, 6] = check_mask
    return result


def example_spectra(observed, passband):
    """Return spatial frames, coherent stacks, joint spectrum and the pass mask."""
    shot = _five_fields(np.asarray(observed)[None]).astype(np.float32)
    mask = passband.mask(*shot.shape[-2:])
    frames, joint = _spectra(shot)
    filtered = fft.ifft(joint[0] * mask, axis=0)[:5]
    stack = np.einsum("t,tyx->yx", WEIGHTS, frames[0])
    filtered_stack = np.einsum("t,tyx->yx", WEIGHTS, filtered)
    return frames[0], stack, filtered_stack, joint[0], mask


def retained_energy(frames, passband):
    """Fraction of total input Fourier power in the passband, pooled over shots."""
    frames = _five_fields(frames)
    mask = passband.mask(*frames.shape[-2:])
    total = retained = 0.0
    for start in range(0, len(frames), 128):
        _, joint = _spectra(frames[start : start + 128].astype(np.float32))
        power = np.abs(joint) ** 2
        total += float(np.sum(power, dtype=np.float64))
        retained += float(np.sum(power, where=mask[None], dtype=np.float64))
    return retained / total if total else None
