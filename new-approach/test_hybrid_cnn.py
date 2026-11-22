import numpy as np

from frequency import Passband, features as fft_features
from hybrid_cnn import HybridCNN, HybridFeatures, make_features
from stim_data import geometry


def test_spatial_features_keep_raw_fft_and_terminal_observations():
    mask = geometry(5).check_mask
    observed = np.zeros((2, 5, 6, 6), dtype=bool)
    observed[0, 3:, 2, 2] = True
    terminal = observed[:, -1, mask].copy()
    terminal[1, 0] = True
    sites = np.full((2, 5, 5), 0.25, dtype=np.float32)
    band = Passband(temporal=0.1, spatial=0.25)
    inputs = make_features(
        observed,
        terminal,
        mask,
        [False, True],
        [0, -3],
        mwpm_weight=[0, 4],
        fft_site_probabilities=sites,
        matching_features=[[0.25], [0.5]],
        passband=band,
    )
    fields = inputs.spatial
    np.testing.assert_array_equal(fields[..., :5].transpose(0, 3, 1, 2), observed)
    np.testing.assert_array_equal(
        fields[..., 5:12].transpose(0, 3, 1, 2), fft_features(observed, mask, band).astype(np.float16)
    )
    np.testing.assert_array_equal(fields[..., 12][:, mask], terminal)
    np.testing.assert_array_equal(fields[..., 13][:, mask], terminal ^ observed[:, -1, mask])
    np.testing.assert_array_equal(fields[0, 2, 2, 14:18], [0, 0, 1, 0])
    np.testing.assert_array_equal(fields[:, :-1, :-1, 18], sites)
    np.testing.assert_array_equal(inputs.matching[:, :2], [[0, 0], [3, 1]])
    np.testing.assert_array_equal(inputs.matching[:, -1], [0.25, 0.5])
    np.testing.assert_array_equal(HybridFeatures.concatenate([inputs.take([0]), inputs.take([1])]).spatial, fields)


def test_soft_target_training_and_checkpoint(tmp_path):
    rng = np.random.default_rng(124)
    mask = geometry(5).check_mask
    shots = 5_000
    observed = (rng.random((shots, 5, *mask.shape)) < 0.20) & mask
    terminal = rng.random((shots, mask.sum())) < 0.25
    parity = rng.random(shots) < 0.40
    margin = rng.choice([0, 0.7, 1.4], size=shots)
    targets = 0.05 + 0.85 * (observed[:, 2, 2, 2] ^ terminal[:, 0])
    inputs = make_features(observed, terminal, mask, parity, margin)
    train, val = inputs.take(slice(0, 4_000)), inputs.take(slice(4_000, None))
    model = HybridCNN(train, seed=125, conv_channels=12, hidden_channels=24)
    initial_kernel = model.kernel.copy()
    initial_scores = model.predict(val, batch_size=128)
    rows = model.fit(
        train,
        targets[:4_000],
        epochs=12,
        batch_size=256,
        lr=0.003,
        seed=126,
        positive_weight=2.0,
        validation=(val, targets[4_000:]),
    )
    scores = model.predict(val, batch_size=128)
    assert np.all((0 < scores) & (scores < 1))
    assert not np.array_equal(model.kernel, initial_kernel)
    assert np.mean((scores - targets[4_000:]) ** 2) < 0.5 * np.mean((initial_scores - targets[4_000:]) ** 2)
    assert rows[-1]["validation_error"] < rows[0]["validation_error"]
    assert rows[-1]["validation_loss"] < rows[0]["validation_loss"]

    checkpoint = tmp_path / "hybrid.npz"
    model.save(checkpoint)
    loaded = HybridCNN.load(checkpoint)
    np.testing.assert_array_equal(loaded.predict(val, batch_size=128), scores)
    assert loaded.predict(val.take(slice(0, 0))).shape == (0,)
