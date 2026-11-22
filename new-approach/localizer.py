"""CPU convolutional localization of data-qubit faults from detector features."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import expit


class Localizer:
    """Apply a shared 4-by-4 check convolution and a per-cell 1-by-1 readout.

    The input has shape (shots, channels, checks_y, checks_x). Each output
    (row, column) corresponds to the data qubit between checks at that index
    and the next row and column. Its receptive field includes those four
    corners and the surrounding ring of checks. An added channel distinguishes
    missing checks in the zero padding from zero-valued features on the grid.
    """

    KERNEL_SIZE = 4
    HIDDEN_CHANNELS = 24

    def __init__(self, channels: int, seed: int) -> None:
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        rng = np.random.default_rng(seed)
        patch_size = (channels + 1) * self.KERNEL_SIZE**2
        self.kernel = rng.standard_normal(
            (patch_size, self.HIDDEN_CHANNELS), dtype=np.float32
        )
        self.kernel *= np.sqrt(2 / patch_size)
        self.bias = np.zeros(self.HIDDEN_CHANNELS, dtype=np.float32)
        self.readout = rng.standard_normal(self.HIDDEN_CHANNELS, dtype=np.float32)
        self.readout *= 0.1 / np.sqrt(self.HIDDEN_CHANNELS)
        self.readout_bias = np.full(1, -3.0, dtype=np.float32)

    def _inputs(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[2:]) < 2:
            raise ValueError(
                f"Expected inputs with shape (shots, {self.channels}, checks_y, checks_x) "
                "and at least two checks per grid axis"
            )
        return x

    def _patches(self, x: np.ndarray) -> np.ndarray:
        shots, channels, height, width = x.shape
        padded = np.zeros((shots, channels + 1, height + 2, width + 2), dtype=np.float32)
        padded[:, :channels, 1:-1, 1:-1] = x
        padded[:, channels, 1:-1, 1:-1] = 1
        patches = sliding_window_view(padded, (self.KERNEL_SIZE, self.KERNEL_SIZE), axis=(2, 3))
        return patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, self.kernel.shape[0])

    def _forward(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = patches @ self.kernel
        hidden += self.bias
        np.maximum(hidden, 0, out=hidden)
        logits = hidden @ self.readout
        logits += self.readout_bias
        return hidden, logits

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
    ) -> list[float]:
        """Train from the current weights with a new Adam optimizer."""
        x = self._inputs(x)
        height, width = x.shape[2:]
        y = np.asarray(y, dtype=np.float32)
        if y.shape != (len(x), height - 1, width - 1) or not len(x):
            raise ValueError("Training targets must match the data-cell grid on a nonempty batch")
        if np.any((y != 0) & (y != 1)):
            raise ValueError("Training targets must be binary")
        if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("epochs, batch_size and lr must be positive")

        positives = np.count_nonzero(y)
        positive_weight = np.float32(
            np.clip((y.size - positives) / max(1, positives), 1, 32)
        )
        # Train against the reweighted odds. Prediction uses the underlying
        # logit so changing the class weight does not shift the inference scale.
        log_weight = np.log(positive_weight)
        params = (self.kernel, self.bias, self.readout, self.readout_bias)
        first_moments = [np.zeros_like(param) for param in params]
        second_moments = [np.zeros_like(param) for param in params]
        rng = np.random.default_rng(seed)
        losses = []
        step = 0

        for _ in range(epochs):
            order = rng.permutation(len(x))
            total_loss = 0.0
            for start in range(0, len(x), batch_size):
                batch = order[start : start + batch_size]
                patches = self._patches(x[batch])
                labels = y[batch].reshape(-1)
                hidden, logits = self._forward(patches)
                logits += log_weight
                sample_weights = np.where(labels == 1, positive_weight, np.float32(1))
                total_loss += np.dot(
                    sample_weights, np.logaddexp(0, np.where(labels == 1, -logits, logits))
                )

                d_logits = (expit(logits) - labels) * sample_weights / labels.size
                d_hidden = d_logits[:, None] * self.readout
                d_hidden *= hidden > 0
                gradients = (
                    patches.T @ d_hidden,
                    d_hidden.sum(axis=0),
                    hidden.T @ d_logits,
                    d_logits.sum(keepdims=True),
                )
                step += 1
                for param, gradient, first, second in zip(
                    params, gradients, first_moments, second_moments
                ):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient**2
                    corrected_first = first / (1 - 0.9**step)
                    corrected_second = second / (1 - 0.999**step)
                    param -= lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            losses.append(float(total_loss / y.size))
        return losses

    def predict(self, x: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
        """Return per-shot, per-cell probabilities using bounded working memory."""
        x = self._inputs(x)
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        height, width = x.shape[2:]
        result = np.empty((len(x), height - 1, width - 1), dtype=np.float32)
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            _, logits = self._forward(self._patches(batch))
            result[start : start + len(batch)] = expit(logits).reshape(-1, height - 1, width - 1)
        return result

    def save(self, path: str | Path) -> None:
        """Write all trainable parameters to the specified checkpoint."""
        with Path(path).open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                channels=self.channels,
                kernel=self.kernel,
                bias=self.bias,
                readout=self.readout,
                readout_bias=self.readout_bias,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Localizer":
        """Load a checkpoint for inference or fine-tuning."""
        with np.load(path, allow_pickle=False) as checkpoint:
            model = cls(channels=int(checkpoint["channels"]), seed=0)
            for name in ("kernel", "bias", "readout", "readout_bias"):
                parameter = getattr(model, name)
                saved = checkpoint[name]
                if saved.shape != parameter.shape:
                    raise ValueError(f"Invalid checkpoint shape for {name}: {saved.shape}")
                parameter[:] = saved
        return model
