"""Shared held-out metrics for learned and traditional traffic generators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


METRIC_KEYS = (
    "size_ks_stat",
    "iat_ks_stat",
    "length_ks_stat",
    "direction_ks_stat",
    "total_bytes_ks_stat",
    "duration_ks_stat",
)


def _mean_lag1(values: np.ndarray, lengths: np.ndarray) -> float | None:
    correlations = []
    for sequence, length in zip(values, lengths):
        active = sequence[:int(length)].astype(np.float64)
        if len(active) <= 2:
            continue
        first = active[:-1] - active[:-1].mean()
        second = active[1:] - active[1:].mean()
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator > 1e-12:
            correlation = float(np.dot(first, second) / denominator)
            if np.isfinite(correlation):
                correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else None


def compute_generation_metrics(
    real_sequences: np.ndarray,
    real_lengths: np.ndarray,
    synthetic_sequences: np.ndarray,
    synthetic_lengths: np.ndarray,
) -> dict:
    """Compute the same distribution and temporal metrics for any generator."""
    real_lengths = np.asarray(real_lengths, dtype=np.int64)
    synthetic_lengths = np.asarray(synthetic_lengths, dtype=np.int64)
    real_sequences = np.asarray(real_sequences)
    synthetic_sequences = np.asarray(synthetic_sequences)

    real_active = (
        np.arange(real_sequences.shape[1])[None, :] < real_lengths[:, None]
    )
    synthetic_active = (
        np.arange(synthetic_sequences.shape[1])[None, :]
        < synthetic_lengths[:, None]
    )
    real_iat_active = real_active.copy()
    synthetic_iat_active = synthetic_active.copy()
    real_iat_active[:, 0] = False
    synthetic_iat_active[:, 0] = False

    real_sizes = real_sequences[:, :, 0][real_active]
    synthetic_sizes = synthetic_sequences[:, :, 0][synthetic_active]
    real_iats = real_sequences[:, :, 1][real_iat_active]
    synthetic_iats = synthetic_sequences[:, :, 1][synthetic_iat_active]

    def per_flow(values: np.ndarray, lengths: np.ndarray, fn) -> np.ndarray:
        return np.asarray([
            fn(values[index, :int(length)])
            for index, length in enumerate(lengths)
        ])

    real_direction = per_flow(real_sequences[:, :, 2], real_lengths, np.mean)
    synthetic_direction = per_flow(
        synthetic_sequences[:, :, 2], synthetic_lengths, np.mean
    )
    real_total_bytes = per_flow(real_sequences[:, :, 0], real_lengths, np.sum)
    synthetic_total_bytes = per_flow(
        synthetic_sequences[:, :, 0], synthetic_lengths, np.sum
    )
    real_duration = np.asarray([
        real_sequences[index, 1:int(length), 1].sum()
        for index, length in enumerate(real_lengths)
    ])
    synthetic_duration = np.asarray([
        synthetic_sequences[index, 1:int(length), 1].sum()
        for index, length in enumerate(synthetic_lengths)
    ])

    def ks(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
        if not len(first) or not len(second):
            return 1.0, 0.0
        result = ks_2samp(first, second)
        return float(result.statistic), float(result.pvalue)

    size_stat, size_p = ks(real_sizes, synthetic_sizes)
    iat_stat, iat_p = ks(real_iats, synthetic_iats)
    length_stat, length_p = ks(real_lengths, synthetic_lengths)
    direction_stat, direction_p = ks(real_direction, synthetic_direction)
    bytes_stat, bytes_p = ks(real_total_bytes, synthetic_total_bytes)
    duration_stat, duration_p = ks(real_duration, synthetic_duration)

    return {
        "size_ks_stat": size_stat,
        "size_p_value": size_p,
        "iat_ks_stat": iat_stat,
        "iat_p_value": iat_p,
        "length_ks_stat": length_stat,
        "length_p_value": length_p,
        "direction_ks_stat": direction_stat,
        "direction_p_value": direction_p,
        "total_bytes_ks_stat": bytes_stat,
        "total_bytes_p_value": bytes_p,
        "duration_ks_stat": duration_stat,
        "duration_p_value": duration_p,
        "real_mean_length": float(real_lengths.mean()),
        "synthetic_mean_length": float(synthetic_lengths.mean()),
        "real_size_lag1": _mean_lag1(real_sequences[:, :, 0], real_lengths),
        "synthetic_size_lag1": _mean_lag1(
            synthetic_sequences[:, :, 0], synthetic_lengths
        ),
        "n_real": int(len(real_lengths)),
        "n_synthetic": int(len(synthetic_lengths)),
    }


def summarize_results(categories: dict) -> dict:
    """Return unweighted category means so rare classes remain visible."""
    return {
        key: float(np.mean([row[key] for row in categories.values()]))
        for key in METRIC_KEYS
    }


def compare_result_files(result_files: dict[str, str | Path], output_dir: str | Path) -> dict:
    """Compare generators over the categories common to every result file."""
    loaded = {
        name: json.loads(Path(path).read_text())
        for name, path in result_files.items()
    }
    category_sets = [set(payload["categories"]) for payload in loaded.values()]
    common_categories = sorted(set.intersection(*category_sets))
    if not common_categories:
        raise ValueError("Generator results have no common evaluated categories")

    summaries = {}
    for name, payload in loaded.items():
        common = {cat: payload["categories"][cat] for cat in common_categories}
        summaries[name] = summarize_results(common)

    winners = {
        metric: min(summaries, key=lambda name: summaries[name][metric])
        for metric in METRIC_KEYS
    }
    comparison = {
        "common_categories": common_categories,
        "category_count": len(common_categories),
        "lower_is_better": list(METRIC_KEYS),
        "summary": summaries,
        "winners": winners,
        "sources": {name: str(Path(path)) for name, path in result_files.items()},
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "comparison.json").write_text(json.dumps(comparison, indent=2))

    short_names = {
        "size_ks_stat": "Size",
        "iat_ks_stat": "IAT",
        "length_ks_stat": "Length",
        "direction_ks_stat": "Direction",
        "total_bytes_ks_stat": "Bytes",
        "duration_ks_stat": "Duration",
    }
    model_names = list(summaries)
    x = np.arange(len(METRIC_KEYS))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, name in enumerate(model_names):
        offset = (index - (len(model_names) - 1) / 2) * width
        ax.bar(
            x + offset,
            [summaries[name][metric] for metric in METRIC_KEYS],
            width,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([short_names[key] for key in METRIC_KEYS])
    ax.set_ylabel("Mean per-category KS statistic (lower is better)")
    ax.set_title(f"Traffic generator comparison ({len(common_categories)} categories)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "comparison.png", dpi=160)
    plt.close(fig)

    header = "| generator | " + " | ".join(short_names[key] for key in METRIC_KEYS) + " |"
    separator = "|:--|" + "--:|" * len(METRIC_KEYS)
    rows = [header, separator]
    for name in model_names:
        values = " | ".join(f"{summaries[name][key]:.3f}" for key in METRIC_KEYS)
        rows.append(f"| {name} | {values} |")
    (output_path / "comparison.md").write_text(
        "# Traffic generator comparison\n\n"
        f"Common held-out categories: {', '.join(common_categories)}. "
        "All values are unweighted category-mean KS statistics; lower is better.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    return comparison
