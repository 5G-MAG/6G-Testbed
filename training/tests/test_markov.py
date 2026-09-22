import numpy as np

from quantization import (
    EmpiricalDequantizer,
    SIZE_BIN_EDGES,
    quantize_iats,
    quantize_sizes,
)
from train_markov import CategoricalMarkovGenerator


def test_empirical_dequantizer_samples_within_bins():
    rng = np.random.default_rng(0)
    n, max_len = 40, 10
    sequences = np.zeros((n, max_len, 3), dtype=np.float32)
    sequences[:, :, 0] = rng.uniform(40, 3000, size=(n, max_len))
    sequences[:, :, 1] = rng.uniform(1e-4, 0.5, size=(n, max_len))
    lengths = rng.integers(2, max_len + 1, size=n)
    categories = rng.integers(0, 2, size=n)

    deq = EmpiricalDequantizer.fit(
        sequences, lengths, categories, np.arange(n), max_len
    )

    bins = quantize_sizes(sequences[:, :, 0])
    bins[0, 5:] = 0  # padding must stay zero
    out = deq.sample("size", bins, categories, np.random.default_rng(1))
    assert out.shape == bins.shape
    assert np.all(out[0, 5:] == 0)
    active = bins > 0
    # Every dequantized value must fall inside its bin's edge interval.
    lo = SIZE_BIN_EDGES[bins[active]]
    hi = SIZE_BIN_EDGES[bins[active] + 1]
    assert np.all(out[active] >= lo - 1e-9)
    assert np.all(out[active] <= hi + 1e-9)

    # Same rng seed reproduces the same draw; round trip via arrays holds.
    again = deq.sample("size", bins, categories, np.random.default_rng(1))
    np.testing.assert_array_equal(out, again)
    restored = EmpiricalDequantizer.from_arrays(deq.to_arrays())
    third = restored.sample("size", bins, categories, np.random.default_rng(1))
    np.testing.assert_array_equal(out, third)
    iat_out = restored.sample(
        "iat", quantize_iats(sequences[:, :, 1]), categories,
        np.random.default_rng(2),
    )
    assert iat_out.shape == (n, max_len)


def test_markov_generator_is_seeded_and_masks_lengths():
    categories, size_bins, iat_bins, max_len = 1, 5, 5, 6
    arrays = {
        "max_len": np.asarray(max_len),
        "supported_category_ids": np.asarray([0]),
        "length_probs": np.asarray([[0, 0, 1, 0, 0, 0]], dtype=float),
        "size_initial": np.asarray([[0, 1, 0, 0, 0]], dtype=float),
        "size_marginal": np.asarray([[0, 0.5, 0.5, 0, 0]], dtype=float),
        "size_transition": np.broadcast_to(
            np.asarray([0, 0.5, 0.5, 0, 0], dtype=float),
            (categories, size_bins, size_bins),
        ).copy(),
        "iat_initial": np.asarray([[1, 0, 0, 0, 0]], dtype=float),
        "iat_marginal": np.asarray([[0.5, 0.5, 0, 0, 0]], dtype=float),
        "iat_transition": np.broadcast_to(
            np.asarray([0.5, 0.5, 0, 0, 0], dtype=float),
            (categories, iat_bins, iat_bins),
        ).copy(),
        "direction_initial": np.asarray([[1, 0]], dtype=float),
        "direction_marginal": np.asarray([[0.5, 0.5]], dtype=float),
        "direction_transition": np.broadcast_to(
            np.asarray([0.5, 0.5], dtype=float),
            (categories, 2, 2),
        ).copy(),
    }
    model = CategoricalMarkovGenerator(arrays, {"streaming": 0})
    first = model.generate(0, 8, order=1, seed=7)
    second = model.generate(0, 8, order=1, seed=7)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(first[1] == 3)
    assert np.all(first[0][:, 3:] == 0)
    assert np.all(first[0][:, :3, 0] > 0)
