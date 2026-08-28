"""Shared quantization codec for the traffic generators.

A flow is represented as a sequence of packets whose size and inter-arrival
time are modelled as categorical distributions over log-spaced bins rather
than as continuous regression targets. Sizes and IATs are heavy tailed and
span several orders of magnitude; a regression formulation collapses toward
the mean and loses the multimodal structure that separates one traffic
category from another.

Every generator that emits bin indices dequantizes through the tables fitted
here, so generator comparisons stay on equal terms. This module holds the
codec on its own, with no model or framework dependency, so a baseline can be
trained and evaluated without importing any particular generator.

Contents:

* :data:`SIZE_BIN_EDGES` / :data:`IAT_BIN_EDGES` and their bin centers.
* :func:`quantize_sizes` / :func:`dequantize_sizes` and the IAT equivalents.
* :class:`EmpiricalDequantizer`, an inverse-CDF within-bin sampler fit on the
  training flows, which replaces bin-center dequantization.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Packet-size quantization
# ---------------------------------------------------------------------------
# Bin edges: log-spaced from 40 to 65536 bytes, plus a zero bin for padding.
# Bin 0 = padding/zero, bins 1..N_SIZE_BINS = actual size buckets.
N_SIZE_BINS = 64

def _build_size_bin_edges(n_bins: int = N_SIZE_BINS) -> np.ndarray:
    """Log-spaced bin edges from 40 to 65536 bytes."""
    return np.concatenate([[0], np.geomspace(40, 65536, n_bins)])

SIZE_BIN_EDGES = _build_size_bin_edges()
SIZE_BIN_CENTERS = (SIZE_BIN_EDGES[:-1] + SIZE_BIN_EDGES[1:]) / 2
SIZE_BIN_CENTERS[0] = 0  # bin 0 = padding

def quantize_sizes(sizes: np.ndarray) -> np.ndarray:
    """Map continuous sizes to bin indices (0 = padding/zero)."""
    return np.digitize(sizes, SIZE_BIN_EDGES[1:]).astype(np.int64)

def dequantize_sizes(bin_indices: np.ndarray) -> np.ndarray:
    """Map bin indices back to representative sizes."""
    return SIZE_BIN_CENTERS[np.clip(bin_indices, 0, N_SIZE_BINS)]

# ---------------------------------------------------------------------------
# IAT quantization (log-spaced, similar approach)
# ---------------------------------------------------------------------------
N_IAT_BINS = 64

def _build_iat_bin_edges(n_bins: int = N_IAT_BINS) -> np.ndarray:
    """Log-spaced bin edges from 1e-6 to 100 seconds."""
    return np.concatenate([[0], np.geomspace(1e-6, 100, n_bins)])

IAT_BIN_EDGES = _build_iat_bin_edges()
IAT_BIN_CENTERS = (IAT_BIN_EDGES[:-1] + IAT_BIN_EDGES[1:]) / 2
IAT_BIN_CENTERS[0] = 0

def quantize_iats(iats: np.ndarray) -> np.ndarray:
    return np.digitize(iats, IAT_BIN_EDGES[1:]).astype(np.int64)

def dequantize_iats(bin_indices: np.ndarray) -> np.ndarray:
    return IAT_BIN_CENTERS[np.clip(bin_indices, 0, N_IAT_BINS)]


# ---------------------------------------------------------------------------
# Empirical within-bin dequantization
# ---------------------------------------------------------------------------
# Bin-center dequantization caps distributional fidelity on its own: the
# synthetic size CDF is a 64-step staircase against a continuous real CDF,
# which alone accounts for a per-category size KS of roughly 0.24 to 0.46 on
# this dataset, regardless of the generator. The dequantizer below replaces
# centers with an inverse-CDF sample of the raw within-bin values observed in
# the clean-profile training flows, per category, falling back to the global
# per-bin distribution and then to the bin center. It is part of the shared
# quantization codec: every generator that emits bin indices (the CVAE and
# the Markov baselines) dequantizes through the same fitted tables, so
# generator comparisons stay fair.

N_DEQUANT_QUANTILES = 64


class EmpiricalDequantizer:
    """Inverse-CDF within-bin sampler fit on training flows."""

    def __init__(self, tables: dict[str, np.ndarray]):
        # Keys: "{channel}_c{category}_b{bin}" and "{channel}_g_b{bin}".
        self.tables = tables

    @staticmethod
    def _grid(values: np.ndarray) -> np.ndarray:
        if len(values) == 1:
            return values.astype(np.float64)
        points = min(len(values), N_DEQUANT_QUANTILES)
        return np.quantile(values.astype(np.float64), np.linspace(0, 1, points))

    @classmethod
    def fit(cls, sequences: np.ndarray, lengths: np.ndarray,
            categories: np.ndarray, indices, max_len: int) -> "EmpiricalDequantizer":
        indices = np.asarray(indices, dtype=np.int64)
        seqs = sequences[indices, :max_len]
        lens = np.minimum(lengths[indices], max_len)
        cats = categories[indices]
        active = np.arange(max_len)[None, :] < lens[:, None]
        iat_active = active.copy()
        iat_active[:, 0] = False  # the first packet's IAT is not meaningful

        tables: dict[str, np.ndarray] = {}
        for channel, raw, mask, quantize in (
            ("size", seqs[:, :, 0], active, quantize_sizes),
            ("iat", seqs[:, :, 1], iat_active, quantize_iats),
        ):
            values = raw[mask]
            bins = quantize(values)
            flat_cats = np.broadcast_to(cats[:, None], mask.shape)[mask]
            for b in np.unique(bins):
                if b == 0:
                    continue  # bin 0 is padding and dequantizes to 0
                in_bin = bins == b
                tables[f"{channel}_g_b{b}"] = cls._grid(values[in_bin])
                for c in np.unique(flat_cats[in_bin]):
                    key = f"{channel}_c{c}_b{b}"
                    tables[key] = cls._grid(values[in_bin & (flat_cats == c)])
        return cls(tables)

    def sample(self, channel: str, bin_indices: np.ndarray,
               categories: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Dequantize (n, len) bin indices for flows with (n,) categories."""
        centers = SIZE_BIN_CENTERS if channel == "size" else IAT_BIN_CENTERS
        bin_indices = np.clip(np.asarray(bin_indices), 0, len(centers) - 1)
        out = centers[bin_indices].astype(np.float64)
        cats = np.broadcast_to(
            np.asarray(categories).reshape(-1, *([1] * (bin_indices.ndim - 1))),
            bin_indices.shape,
        )
        for c in np.unique(cats):
            for b in np.unique(bin_indices[cats == c]):
                if b == 0:
                    continue
                grid = self.tables.get(
                    f"{channel}_c{c}_b{b}", self.tables.get(f"{channel}_g_b{b}")
                )
                if grid is None or len(grid) == 0:
                    continue
                sel = (cats == c) & (bin_indices == b)
                if len(grid) == 1:
                    out[sel] = grid[0]
                else:
                    u = rng.random(int(sel.sum()))
                    out[sel] = np.interp(u, np.linspace(0, 1, len(grid)), grid)
        return out.astype(np.float32)

    def to_arrays(self, prefix: str = "deq_") -> dict[str, np.ndarray]:
        return {f"{prefix}{key}": np.asarray(value)
                for key, value in self.tables.items()}

    @classmethod
    def from_arrays(cls, arrays, prefix: str = "deq_") -> "EmpiricalDequantizer | None":
        tables = {key[len(prefix):]: np.asarray(arrays[key])
                  for key in arrays if key.startswith(prefix)}
        return cls(tables) if tables else None


