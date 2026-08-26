"""Traditional category-conditioned Markov baselines for traffic generation.

Two baselines share the same fitted model:

* order 0 samples packet size, IAT, and direction independently from their
  category marginals;
* order 1 samples each channel from a first-order transition matrix with an
  empirical-marginal backoff for sparse states.

Both sample sequence length from the empirical per-category distribution and
are evaluated on the exact clean-profile held-out split used by the CVAE.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from dataset import CLEAN_PROFILES
from generator_metrics import compare_result_files, compute_generation_metrics
from quantization import (
    EmpiricalDequantizer,
    IAT_BIN_CENTERS,
    N_IAT_BINS,
    N_SIZE_BINS,
    SIZE_BIN_CENTERS,
    quantize_iats,
    quantize_sizes,
)

logger = logging.getLogger(__name__)


def _probabilities(counts: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
    values = counts.astype(np.float64) + smoothing
    total = values.sum()
    if total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


def _transition_probabilities(
    counts: np.ndarray,
    marginal: np.ndarray,
    backoff_strength: float,
) -> np.ndarray:
    """Normalize transition rows with a category-marginal Dirichlet prior."""
    backed = counts.astype(np.float64) + backoff_strength * marginal[None, :]
    totals = backed.sum(axis=1, keepdims=True)
    return np.divide(
        backed,
        totals,
        out=np.broadcast_to(marginal, backed.shape).copy(),
        where=totals > 0,
    )


class CategoricalMarkovGenerator:
    """Zero/first-order Markov generator over quantized traffic channels."""

    def __init__(self, arrays: dict[str, np.ndarray], category_ids: dict[str, int]):
        self.arrays = arrays
        self.category_ids = category_ids
        self.max_len = int(np.asarray(arrays["max_len"]).item())
        self.supported_category_ids = set(
            int(value) for value in arrays["supported_category_ids"].tolist()
        )
        # Shared empirical within-bin dequantizer (same codec as the CVAE);
        # older fitted models fall back to bin centers.
        self.dequantizer = EmpiricalDequantizer.from_arrays(arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CategoricalMarkovGenerator":
        payload = np.load(path, allow_pickle=False)
        arrays = {key: payload[key] for key in payload.files if key != "category_ids_json"}
        category_ids = json.loads(str(payload["category_ids_json"].item()))
        return cls(arrays, category_ids)

    def sequence_log_prob(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        categories: np.ndarray,
        order: int,
        floor: float = 1e-12,
    ) -> dict[str, np.ndarray]:
        """Per-packet log probability of observed bins under the fitted tables.

        These generators are next-packet predictors already, so they are the
        natural likelihood baseline for a learned predictor. Position 0 is
        scored against the fitted initial distribution for both orders, which is
        the same information a predictor has at an empty history.

        Returns per-channel arrays shaped (n, max_len), with NaN where the
        packet is padding or the category was never fitted.
        """
        sequences = np.asarray(sequences)
        lengths = np.asarray(lengths, dtype=np.int64)
        categories = np.asarray(categories, dtype=np.int64)
        n = sequences.shape[0]
        max_len = int(min(sequences.shape[1], self.max_len))
        channels = {
            "size": quantize_sizes(sequences[:, :max_len, 0]),
            "iat": quantize_iats(sequences[:, :max_len, 1]),
            "direction": sequences[:, :max_len, 2].astype(np.int64),
        }

        out = {name: np.full((n, max_len), np.nan) for name in channels}
        for name, bins in channels.items():
            initial = self.arrays[f"{name}_initial"]
            marginal = self.arrays[f"{name}_marginal"]
            transition = self.arrays[f"{name}_transition"]
            for i in range(n):
                cat = int(categories[i])
                if cat not in self.supported_category_ids:
                    continue
                limit = int(min(lengths[i], max_len))
                for t in range(limit):
                    if t == 0:
                        row = initial[cat]
                    elif order == 1:
                        row = transition[cat, int(bins[i, t - 1])]
                    else:
                        row = marginal[cat]
                    out[name][i, t] = np.log(max(float(row[int(bins[i, t])]), floor))
        return out

    def _sample_chain(
        self,
        rng: np.random.Generator,
        initial: np.ndarray,
        marginal: np.ndarray,
        transitions: np.ndarray,
        length: int,
        order: int,
    ) -> np.ndarray:
        if length <= 0:
            return np.empty(0, dtype=np.int64)
        result = np.empty(length, dtype=np.int64)
        result[0] = rng.choice(len(initial), p=initial)
        for step in range(1, length):
            probabilities = transitions[result[step - 1]] if order == 1 else marginal
            result[step] = rng.choice(len(probabilities), p=probabilities)
        return result

    def generate(
        self,
        category_id: int,
        n_samples: int,
        order: int = 1,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        if order not in (0, 1):
            raise ValueError("Markov order must be 0 or 1")
        if category_id not in self.supported_category_ids:
            raise ValueError(f"Category id {category_id} has no training sequences")

        rng = np.random.default_rng(seed)
        length_probs = self.arrays["length_probs"][category_id]
        lengths = rng.choice(
            np.arange(1, self.max_len + 1), size=n_samples, p=length_probs
        ).astype(np.int64)
        result = np.zeros((n_samples, self.max_len, 3), dtype=np.float32)
        size_bin_matrix = np.zeros((n_samples, self.max_len), dtype=np.int64)
        iat_bin_matrix = np.zeros((n_samples, self.max_len), dtype=np.int64)

        for sample_index, length in enumerate(lengths):
            size_bins = self._sample_chain(
                rng,
                self.arrays["size_initial"][category_id],
                self.arrays["size_marginal"][category_id],
                self.arrays["size_transition"][category_id],
                int(length),
                order,
            )
            directions = self._sample_chain(
                rng,
                self.arrays["direction_initial"][category_id],
                self.arrays["direction_marginal"][category_id],
                self.arrays["direction_transition"][category_id],
                int(length),
                order,
            )

            iat_bins = np.zeros(int(length), dtype=np.int64)
            if length > 1:
                iat_bins[1:] = self._sample_chain(
                    rng,
                    self.arrays["iat_initial"][category_id],
                    self.arrays["iat_marginal"][category_id],
                    self.arrays["iat_transition"][category_id],
                    int(length) - 1,
                    order,
                )

            size_bin_matrix[sample_index, :length] = size_bins
            iat_bin_matrix[sample_index, :length] = iat_bins
            result[sample_index, :length, 2] = directions

        flow_categories = np.full(n_samples, category_id, dtype=np.int64)
        if self.dequantizer is not None:
            result[:, :, 0] = self.dequantizer.sample(
                "size", size_bin_matrix, flow_categories, rng
            )
            result[:, :, 1] = self.dequantizer.sample(
                "iat", iat_bin_matrix, flow_categories, rng
            )
        else:
            result[:, :, 0] = SIZE_BIN_CENTERS[size_bin_matrix]
            result[:, :, 1] = IAT_BIN_CENTERS[iat_bin_matrix]

        return result, lengths


def fit_markov(
    data_dir: str,
    model_path: str,
    backoff_strength: float = 5.0,
) -> CategoricalMarkovGenerator:
    data_path = Path(data_dir)
    sequence_data = np.load(data_path / "sequences.npz")
    sequences = sequence_data["sequences"]
    categories = sequence_data["categories"].astype(np.int64)
    profiles = sequence_data["profiles"]
    lengths = sequence_data["lengths"].astype(np.int64)
    with open(data_path / "labels.json") as handle:
        labels = json.load(handle)
    with open(data_path / "splits.json") as handle:
        splits = json.load(handle)

    max_len = sequences.shape[1]
    n_categories = max(labels["category_ids"].values()) + 1
    clean_ids = {
        labels["profile_ids"][name]
        for name in labels.get("clean_profiles", CLEAN_PROFILES)
        if name in labels["profile_ids"]
    }
    train_idx = np.asarray([
        index for index in splits["train"] if profiles[index] in clean_ids
    ], dtype=np.int64)
    if not len(train_idx):
        raise ValueError("No clean-profile training sequences for Markov baseline")

    size_bins = quantize_sizes(sequences[:, :, 0])
    iat_bins = quantize_iats(sequences[:, :, 1])

    shape_size = (n_categories, N_SIZE_BINS + 1)
    shape_iat = (n_categories, N_IAT_BINS + 1)
    size_initial_counts = np.zeros(shape_size)
    size_counts = np.zeros(shape_size)
    size_transition_counts = np.zeros((*shape_size, N_SIZE_BINS + 1))
    iat_initial_counts = np.zeros(shape_iat)
    iat_counts = np.zeros(shape_iat)
    iat_transition_counts = np.zeros((*shape_iat, N_IAT_BINS + 1))
    direction_initial_counts = np.zeros((n_categories, 2))
    direction_counts = np.zeros((n_categories, 2))
    direction_transition_counts = np.zeros((n_categories, 2, 2))
    length_counts = np.zeros((n_categories, max_len))

    supported = sorted(int(value) for value in np.unique(categories[train_idx]))
    for index in train_idx:
        category = int(categories[index])
        length = int(np.clip(lengths[index], 1, max_len))
        sizes = size_bins[index, :length]
        iats = iat_bins[index, :length]
        directions = sequences[index, :length, 2].astype(np.int64).clip(0, 1)

        length_counts[category, length - 1] += 1
        size_initial_counts[category, sizes[0]] += 1
        direction_initial_counts[category, directions[0]] += 1
        np.add.at(size_counts[category], sizes, 1)
        np.add.at(direction_counts[category], directions, 1)
        if length > 1:
            np.add.at(size_transition_counts[category], (sizes[:-1], sizes[1:]), 1)
            np.add.at(
                direction_transition_counts[category],
                (directions[:-1], directions[1:]),
                1,
            )
            active_iats = iats[1:]
            iat_initial_counts[category, active_iats[0]] += 1
            np.add.at(iat_counts[category], active_iats, 1)
            if len(active_iats) > 1:
                np.add.at(
                    iat_transition_counts[category],
                    (active_iats[:-1], active_iats[1:]),
                    1,
                )

    # Same within-bin dequantization codec as the CVAE, fit on the same
    # clean-profile training partition, so the comparison stays fair.
    dequantizer = EmpiricalDequantizer.fit(
        sequences, lengths, categories, train_idx, max_len
    )

    arrays: dict[str, np.ndarray] = {
        "max_len": np.asarray(max_len, dtype=np.int64),
        "supported_category_ids": np.asarray(supported, dtype=np.int64),
        **dequantizer.to_arrays(),
    }
    for category in range(n_categories):
        # Active packet sizes must never sample padding bin zero.
        size_counts[category, 0] = 0
        size_initial_counts[category, 0] = 0
        size_marginal = _probabilities(size_counts[category])
        size_marginal[0] = 0
        size_marginal /= size_marginal.sum()
        size_initial = _probabilities(size_initial_counts[category])
        size_initial[0] = 0
        size_initial /= size_initial.sum()

        iat_marginal = _probabilities(iat_counts[category])
        iat_initial = _probabilities(iat_initial_counts[category])
        direction_marginal = _probabilities(direction_counts[category])
        direction_initial = _probabilities(direction_initial_counts[category])

        for name, value, width in (
            ("size_marginal", size_marginal, N_SIZE_BINS + 1),
            ("size_initial", size_initial, N_SIZE_BINS + 1),
            ("iat_marginal", iat_marginal, N_IAT_BINS + 1),
            ("iat_initial", iat_initial, N_IAT_BINS + 1),
            ("direction_marginal", direction_marginal, 2),
            ("direction_initial", direction_initial, 2),
        ):
            arrays.setdefault(name, np.zeros((n_categories, width)))[category] = value

        arrays.setdefault(
            "size_transition",
            np.zeros((n_categories, N_SIZE_BINS + 1, N_SIZE_BINS + 1)),
        )[category] = _transition_probabilities(
            size_transition_counts[category], size_marginal, backoff_strength
        )
        arrays.setdefault(
            "iat_transition",
            np.zeros((n_categories, N_IAT_BINS + 1, N_IAT_BINS + 1)),
        )[category] = _transition_probabilities(
            iat_transition_counts[category], iat_marginal, backoff_strength
        )
        arrays.setdefault(
            "direction_transition", np.zeros((n_categories, 2, 2))
        )[category] = _transition_probabilities(
            direction_transition_counts[category],
            direction_marginal,
            backoff_strength,
        )
        arrays.setdefault(
            "length_probs", np.zeros((n_categories, max_len))
        )[category] = _probabilities(length_counts[category], smoothing=0.0)

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **arrays,
        category_ids_json=np.asarray(json.dumps(labels["category_ids"])),
    )
    metadata = {
        "model": "category-conditioned categorical Markov",
        "orders": [0, 1],
        "backoff_strength": backoff_strength,
        "n_train": int(len(train_idx)),
        "max_len": int(max_len),
        "category_ids": labels["category_ids"],
        "supported_category_ids": supported,
        "split_strategy": labels.get("split_strategy", {}),
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    logger.info("Fitted Markov baseline on %d clean training flows", len(train_idx))
    return CategoricalMarkovGenerator(arrays, labels["category_ids"])


def evaluate_markov(
    model: CategoricalMarkovGenerator,
    data_dir: str,
    output_dir: str,
    synthetic_dir: str,
    seed: int = 42,
) -> dict[int, Path]:
    data_path = Path(data_dir)
    sequence_data = np.load(data_path / "sequences.npz")
    sequences = sequence_data["sequences"][:, :model.max_len]
    categories = sequence_data["categories"].astype(np.int64)
    profiles = sequence_data["profiles"]
    lengths = np.minimum(sequence_data["lengths"], model.max_len).astype(np.int64)
    with open(data_path / "labels.json") as handle:
        labels = json.load(handle)
    with open(data_path / "splits.json") as handle:
        splits = json.load(handle)

    clean_ids = {
        labels["profile_ids"][name]
        for name in labels.get("clean_profiles", CLEAN_PROFILES)
        if name in labels["profile_ids"]
    }
    test_idx = [index for index in splits["test"] if profiles[index] in clean_ids]
    id_to_category = {value: key for key, value in labels["category_ids"].items()}
    outputs = {}

    for order in (0, 1):
        category_results = {}
        order_synthetic = Path(synthetic_dir) / f"markov_order{order}"
        order_synthetic.mkdir(parents=True, exist_ok=True)
        for category_id in sorted(set(categories[test_idx])):
            category_id = int(category_id)
            if category_id not in model.supported_category_ids:
                continue
            category_name = id_to_category[category_id]
            category_idx = np.asarray([
                index for index in test_idx if categories[index] == category_id
            ], dtype=np.int64)

            generated, generated_lengths = model.generate(
                category_id, 100, order=order, seed=seed + category_id
            )
            np.savez_compressed(
                order_synthetic / f"{category_name}.npz",
                sequences=generated,
                lengths=generated_lengths,
            )
            if len(category_idx) < 2:
                continue
            # Same synthetic sample count as the CVAE evaluation, so the
            # KS statistics carry the same sampling variance for every
            # generator.
            benchmark_sequences, benchmark_lengths = model.generate(
                category_id,
                max(512, len(category_idx)),
                order=order,
                seed=seed + category_id,
            )
            category_results[category_name] = compute_generation_metrics(
                sequences[category_idx],
                lengths[category_idx],
                benchmark_sequences,
                benchmark_lengths,
            )

        order_path = Path(output_dir) / f"order{order}"
        order_path.mkdir(parents=True, exist_ok=True)
        result_path = order_path / "ks_tests.json"
        result_path.write_text(json.dumps({
            "model": f"Markov order {order}",
            "seed": seed,
            "split_strategy": labels.get("split_strategy", {}),
            "categories": category_results,
        }, indent=2))
        outputs[order] = result_path
        logger.info(
            "Evaluated Markov order %d over %d held-out categories",
            order,
            len(category_results),
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Markov traffic baselines")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/markov/model.npz")
    parser.add_argument("--results-dir", default="results/generator_markov")
    parser.add_argument("--synthetic-dir", default="synthetic")
    parser.add_argument("--cvae-results", default="results/generator/ks_tests.json")
    parser.add_argument("--previous-cvae-results", nargs="*", default=[])
    parser.add_argument("--comparison-dir", default="results/generator_comparison")
    parser.add_argument("--backoff-strength", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model = fit_markov(args.data_dir, args.model_path, args.backoff_strength)
    results = evaluate_markov(
        model,
        args.data_dir,
        args.results_dir,
        args.synthetic_dir,
        args.seed,
    )
    def cvae_label(path: str | Path) -> str:
        version = json.loads(Path(path).read_text()).get("model_version", "?")
        return f"CVAE v{version}"

    result_files = {
        cvae_label(args.cvae_results): args.cvae_results,
        "Markov order 0": results[0],
        "Markov order 1": results[1],
    }
    for previous in args.previous_cvae_results:
        if Path(previous).exists():
            result_files = {cvae_label(previous): previous, **result_files}
    comparison = compare_result_files(result_files, args.comparison_dir)
    print("\n=== Mean per-category KS comparison (lower is better) ===")
    for name, summary in comparison["summary"].items():
        print(
            f"{name:16s} size={summary['size_ks_stat']:.3f} "
            f"iat={summary['iat_ks_stat']:.3f} "
            f"length={summary['length_ks_stat']:.3f} "
            f"direction={summary['direction_ks_stat']:.3f}"
        )
    print(f"\nComparison: {Path(args.comparison_dir) / 'comparison.md'}")


if __name__ == "__main__":
    main()
