import numpy as np

from localizer import Localizer


def _corner_patterns(shots, seed, noise=0.12):
    rng = np.random.default_rng(seed)
    targets = rng.random((shots, 3, 3)) < 0.08
    inputs = rng.normal(0, noise, (shots, 2, 4, 4)).astype(np.float32)
    inputs[:, 0, :-1, :-1] += targets
    inputs[:, 1, 1:, 1:] += targets
    return inputs, targets


def test_localizer_learns_reproducibly():
    train_inputs, train_targets = _corner_patterns(768, seed=9)
    test_inputs, test_targets = _corner_patterns(128, seed=10)
    first = Localizer(channels=2, seed=11)
    second = Localizer(channels=2, seed=11)
    training = dict(epochs=6, batch_size=32, lr=0.01, seed=12)

    losses = first.fit(train_inputs, train_targets, **training)
    second.fit(train_inputs, train_targets, **training)
    predicted = first.predict(test_inputs, batch_size=33)

    assert len(losses) == training["epochs"]
    assert losses[-1] < 0.3 * losses[0]
    assert predicted.shape == test_targets.shape
    assert predicted.dtype == np.float32
    assert np.all((0 <= predicted) & (predicted <= 1))
    assert predicted[test_targets].mean() > 0.7
    assert predicted[~test_targets].mean() < 0.15
    np.testing.assert_array_equal(second.predict(test_inputs, batch_size=33), predicted)


def test_checkpoint_preserves_predictions_and_supports_independent_fine_tuning(tmp_path):
    train_inputs, train_targets = _corner_patterns(128, seed=13)
    noisy_inputs, noisy_targets = _corner_patterns(128, seed=14, noise=0.35)
    original = Localizer(channels=2, seed=15)
    original.fit(train_inputs, train_targets, epochs=2, batch_size=32, lr=0.01, seed=16)
    before = original.predict(noisy_inputs)
    checkpoint = tmp_path / "pretrained.npz"
    original.save(checkpoint)

    adapted = Localizer.load(checkpoint)
    np.testing.assert_array_equal(adapted.predict(noisy_inputs), before)
    adapted.fit(noisy_inputs, noisy_targets, epochs=2, batch_size=32, lr=0.01, seed=17)

    assert np.max(np.abs(adapted.predict(noisy_inputs) - before)) > 0.01
    np.testing.assert_array_equal(original.predict(noisy_inputs), before)
    np.testing.assert_array_equal(Localizer.load(checkpoint).predict(noisy_inputs), before)
