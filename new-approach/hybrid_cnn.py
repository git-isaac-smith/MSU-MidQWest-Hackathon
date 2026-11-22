"""Convolutional estimate of a matching decoder's logical error probability."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from frequency import Passband, features as fft_features


@dataclass(frozen=True)
class HybridFeatures:
    """Public spatial and matching inputs, separate from logical training targets."""

    spatial: np.ndarray
    matching: np.ndarray
    check_mask: np.ndarray

    def __len__(self):
        return len(self.spatial)

    def take(self, selection):
        return HybridFeatures(self.spatial[selection], self.matching[selection], self.check_mask)

    @classmethod
    def concatenate(cls, blocks):
        blocks = tuple(blocks)
        if not blocks or any(not np.array_equal(block.check_mask, blocks[0].check_mask) for block in blocks):
            raise ValueError("Need at least one feature block on a common check grid")
        return cls(
            np.concatenate([block.spatial for block in blocks]),
            np.concatenate([block.matching for block in blocks]),
            blocks[0].check_mask,
        )


def _shot_column(values, shots, name):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (shots,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain one finite value per shot")
    return values


def make_features(
    observed,
    terminal,
    check_mask,
    mwpm_parity,
    mwpm_margin,
    *,
    mwpm_weight=None,
    fft_site_probabilities=None,
    matching_features=None,
    passband=None,
):
    """Encode measurements and matching outputs without using any logical labels.

    Terminal checks are in row-major active-check order. The margin is the
    gap between the two logical sectors, in matching's log-odds cost units.
    Optional scalar columns must use a fixed scale across the data splits.
    """
    raw = np.asarray(observed)
    mask = np.asarray(check_mask, dtype=bool)
    if raw.ndim != 4 or raw.shape[1] != 5 or raw.shape[2:] != mask.shape or not mask.any():
        raise ValueError("Expected five observed grids on a nonempty check mask")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("Observed grids must be binary")
    raw = raw.astype(bool, copy=False)
    shots, _, height, width = raw.shape
    final = np.asarray(terminal)
    if final.shape != (shots, np.count_nonzero(mask)) or np.any((final != 0) & (final != 1)):
        raise ValueError("Expected one binary terminal parity per active check per shot")
    final = final.astype(bool, copy=False)
    parity = _shot_column(mwpm_parity, shots, "Matching parity")
    if np.any((parity != 0) & (parity != 1)):
        raise ValueError("Matching parity must be binary")
    margin = _shot_column(mwpm_margin, shots, "Matching margin")

    # Keep compact resident arrays. Arithmetic and the convolution run in float32.
    filtered = fft_features(raw, mask, passband if passband is not None else Passband())
    events = raw[:, 1:] ^ raw[:, :-1]
    terminal_events = raw[:, -1, mask] ^ final
    spatial = np.zeros((shots, height, width, 18 + (fft_site_probabilities is not None)), dtype=np.float16)
    spatial[..., :5] = raw.transpose(0, 2, 3, 1)
    spatial[..., 5:12] = filtered.transpose(0, 2, 3, 1)
    spatial[..., 12][:, mask] = final
    spatial[..., 13][:, mask] = terminal_events
    spatial[..., 14:18] = events.transpose(0, 2, 3, 1)
    if fft_site_probabilities is not None:
        sites = np.asarray(fft_site_probabilities, dtype=np.float32)
        if sites.shape != (shots, height - 1, width - 1) or not np.isfinite(sites).all():
            raise ValueError("Expected one finite FFT probability per data site per shot")
        if np.any((sites < 0) | (sites > 1)):
            raise ValueError("FFT site probabilities must be between zero and one")
        spatial[:, :-1, :-1, 18] = sites

    activity = np.column_stack((
        raw[:, 0].sum(axis=(1, 2)),
        events.sum(axis=(2, 3)),
        terminal_events.sum(axis=1),
        final.sum(axis=1),
    )).astype(np.float32)
    matching = [
        np.abs(margin),
        parity,
        np.sign(margin) * np.log1p(np.abs(margin)),
        np.log1p(activity),
    ]
    if mwpm_weight is not None:
        weight = _shot_column(mwpm_weight, shots, "Matching weight")
        matching.append(np.sign(weight) * np.log1p(np.abs(weight)))
    if matching_features is not None:
        extra = np.asarray(matching_features, dtype=np.float32)
        if extra.ndim != 2 or len(extra) != shots or not np.isfinite(extra).all():
            raise ValueError("Additional matching features must be a finite shot-by-feature array")
        matching.append(extra)
    return HybridFeatures(spatial, np.column_stack(matching), mask.copy())


class HybridCNN:
    """A shared 3-by-3 spatial convolution and a position-aware logical readout.

    The convolution is evaluated at active checks. A dense layer reads the
    resulting local features in fixed spatial order together with matching
    features, so it can distinguish the boundaries of the logical operator.
    A fixed log-odds term starts from the matching cost gap. The CNN learns
    the residual needed for probability mass spread across fault histories.
    """

    _PARAMETERS = ("kernel", "conv_bias", "dense", "dense_bias", "readout", "skip", "bias")

    def __init__(self, inputs: HybridFeatures, *, seed=0, conv_channels=24, hidden_channels=64):
        self.check_mask = np.array(inputs.check_mask, dtype=bool, copy=True)
        self.channels = inputs.spatial.shape[-1]
        self.matching_columns = inputs.matching.shape[-1]
        self._check_inputs(inputs)
        if self.channels < 1 or self.matching_columns < 2 or conv_channels < 1 or hidden_channels < 1:
            raise ValueError("The number of spatial, matching, and hidden channels must be positive")

        width = self.check_mask.shape[1] + 2
        y, x = np.nonzero(self.check_mask)
        center = (y + 1) * width + (x + 1)
        offsets = np.array([-width - 1, -width, -width + 1, -1, 0, 1, width - 1, width, width + 1])
        self._neighbors = center[:, None] + offsets

        rng = np.random.default_rng(seed)
        patches = self._neighbors.shape[1] * self.channels
        dense_inputs = len(center) * conv_channels + self.matching_columns - 1
        self.kernel = rng.standard_normal((patches, conv_channels), dtype=np.float32) * np.float32(np.sqrt(2 / patches))
        self.conv_bias = np.zeros(conv_channels, dtype=np.float32)
        self.dense = rng.standard_normal((dense_inputs, hidden_channels), dtype=np.float32) * np.float32(np.sqrt(2 / dense_inputs))
        self.dense_bias = np.zeros(hidden_channels, dtype=np.float32)
        self.readout = rng.standard_normal(hidden_channels, dtype=np.float32) * np.float32(0.05 / np.sqrt(hidden_channels))
        self.skip = np.zeros(self.matching_columns - 1, dtype=np.float32)
        self.bias = np.zeros(1, dtype=np.float32)

    def _check_inputs(self, inputs):
        if self.check_mask.ndim != 2 or not self.check_mask.any():
            raise ValueError("Expected a two-dimensional check mask with at least one active check")
        if (
            inputs.spatial.shape != (len(inputs), *self.check_mask.shape, self.channels)
            or inputs.matching.shape != (len(inputs), self.matching_columns)
            or not np.array_equal(inputs.check_mask, self.check_mask)
        ):
            raise ValueError("Spatial data, matching features, and check geometry must match the model")

    def _forward(self, spatial, matching):
        shots, height, width, channels = spatial.shape
        padded = np.zeros((shots, height + 2, width + 2, channels), dtype=np.float32)
        padded[:, 1:-1, 1:-1] = spatial
        patches = np.take(padded.reshape(shots, (height + 2) * (width + 2), channels), self._neighbors, axis=1)
        patches = patches.reshape(-1, self.kernel.shape[0])
        convolution = patches @ self.kernel + self.conv_bias
        np.maximum(convolution, 0, out=convolution)

        covariates = matching[:, 1:]
        combined = np.concatenate((convolution.reshape(shots, self.dense.shape[0] - self.skip.size), covariates), axis=1)
        hidden = combined @ self.dense + self.dense_bias
        np.maximum(hidden, 0, out=hidden)
        logits = hidden @ self.readout + covariates @ self.skip + self.bias - matching[:, 0]
        return patches, convolution, combined, hidden, logits

    @staticmethod
    def _targets(values, shots):
        target = _shot_column(values, shots, "Training targets")
        if np.any((target < 0) | (target > 1)):
            raise ValueError("Logical error targets must be between zero and one")
        return target

    def fit(
        self,
        inputs,
        targets,
        *,
        epochs,
        batch_size=512,
        lr=0.001,
        seed=0,
        sample_weight=None,
        positive_weight=1.0,
        validation=None,
        progress=None,
        weight_decay=1e-5,
    ):
        """Train with binary cross entropy on hard outcomes or soft targets.

        Positive weighting tilts the odds only inside the training objective.
        Validation, when supplied, uses unweighted probabilities and targets.
        `progress(epoch, metrics)` is called after each one-based epoch.
        """
        self._check_inputs(inputs)
        shots = len(inputs)
        target = self._targets(targets, shots)
        if not shots or epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Training data, epochs, batch size, and learning rate must be positive")
        if not np.isfinite(positive_weight) or positive_weight <= 0:
            raise ValueError("Positive weight must be finite and positive")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        weights = np.ones(shots, dtype=np.float32) if sample_weight is None else _shot_column(sample_weight, shots, "Sample weights")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Sample weights must be nonnegative with a positive sum")
        if validation is not None:
            val_inputs, val_targets = validation
            self._check_inputs(val_inputs)
            val_targets = self._targets(val_targets, len(val_inputs))
            if not len(val_targets):
                raise ValueError("Validation data must be nonempty")

        parameters = [getattr(self, name) for name in self._PARAMETERS]
        first = [np.zeros_like(parameter) for parameter in parameters]
        second = [np.zeros_like(parameter) for parameter in parameters]
        positives = np.float32(positive_weight) * target
        negatives = 1 - target
        shift = np.float32(np.log(positive_weight))
        normalizer = float(np.dot(weights, positives + negatives))
        decay = np.float32(weight_decay)
        rng = np.random.default_rng(seed)
        history = []
        step = 0

        for epoch in range(1, epochs + 1):
            order = rng.permutation(shots)
            loss = 0.0
            for start in range(0, shots, batch_size):
                selected = order[start : start + batch_size]
                matching = inputs.matching[selected]
                patches, convolution, combined, hidden, logits = self._forward(inputs.spatial[selected], matching)
                logits += shift
                pos, neg = positives[selected], negatives[selected]
                batch_weights = weights[selected]
                loss += float(batch_weights @ (pos * np.logaddexp(0, -logits) + neg * np.logaddexp(0, logits)))

                output_gradient = (expit(logits) * (pos + neg) - pos) * batch_weights * np.float32(shots / (len(selected) * normalizer))
                hidden_gradient = output_gradient[:, None] * self.readout
                hidden_gradient *= hidden > 0
                combined_gradient = hidden_gradient @ self.dense.T
                conv_gradient = combined_gradient[:, :self.dense.shape[0] - self.skip.size].reshape(convolution.shape)
                conv_gradient *= convolution > 0
                gradients = (
                    patches.T @ conv_gradient + decay * self.kernel,
                    conv_gradient.sum(axis=0),
                    combined.T @ hidden_gradient + decay * self.dense,
                    hidden_gradient.sum(axis=0),
                    hidden.T @ output_gradient + decay * self.readout,
                    matching[:, 1:].T @ output_gradient + decay * self.skip,
                    output_gradient.sum(keepdims=True),
                )
                step += 1
                rate = np.float32(lr * np.sqrt(1 - 0.999**step) / (1 - 0.9**step))
                epsilon = np.float32(1e-8 * np.sqrt(1 - 0.999**step))
                for parameter, gradient, mean, mean_square in zip(parameters, gradients, first, second):
                    mean *= 0.9
                    mean += 0.1 * gradient
                    mean_square *= 0.999
                    mean_square += 0.001 * gradient**2
                    parameter -= rate * mean / (np.sqrt(mean_square) + epsilon)

            metrics = {"train_loss": loss / normalizer}
            if validation is not None:
                val_logits = self._logits(val_inputs, batch_size)
                metrics["validation_loss"] = float(np.mean(np.logaddexp(0, val_logits) - val_targets * val_logits))
                metrics["validation_error"] = float(np.mean(np.where(val_logits > 0, 1 - val_targets, val_targets)))
            history.append(metrics)
            if progress is not None:
                progress(epoch, metrics)
        return history

    def _logits(self, inputs, batch_size):
        self._check_inputs(inputs)
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        output = np.empty(len(inputs), dtype=np.float32)
        for start in range(0, len(inputs), batch_size):
            batch = slice(start, start + batch_size)
            output[batch] = self._forward(inputs.spatial[batch], inputs.matching[batch])[-1]
        return output

    def predict(self, inputs, *, batch_size=1024):
        """Return the predicted probability that ordinary matching is wrong."""
        return expit(self._logits(inputs, batch_size))

    def save(self, path):
        """Write the model geometry and trainable parameters to a NumPy checkpoint."""
        with Path(path).open("wb") as checkpoint:
            np.savez_compressed(
                checkpoint,
                check_mask=self.check_mask,
                channels=self.channels,
                matching_columns=self.matching_columns,
                **{name: getattr(self, name) for name in self._PARAMETERS},
            )

    @classmethod
    def load(cls, path):
        """Load all parameters for inference or further training."""
        with np.load(path, allow_pickle=False) as saved:
            mask = saved["check_mask"]
            inputs = HybridFeatures(
                np.empty((0, *mask.shape, int(saved["channels"])), dtype=np.float16),
                np.empty((0, int(saved["matching_columns"])), dtype=np.float32),
                mask,
            )
            model = cls(inputs, conv_channels=len(saved["conv_bias"]), hidden_channels=len(saved["dense_bias"]))
            for name in cls._PARAMETERS:
                parameter = getattr(model, name)
                value = saved[name]
                if value.shape != parameter.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint array for {name}")
                parameter[:] = value
        return model
