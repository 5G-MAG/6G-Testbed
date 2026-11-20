#!/usr/bin/env python3
"""Create explicitly synthetic media fixtures for integration checks.

These are not representative SA1 workload measurements. Supply a licensed
photo burst with documented encodings for that campaign.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(directory, width=512, height=288):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose an empty output directory to preserve existing assets")
    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(26870)
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    linear = np.stack([x, y, 0.5 + 0.3 * np.sin(25 * x) * np.cos(25 * y)], axis=2)
    manifest = {"workload_kind": "synthetic_fixture", "seed": 26870, "images": []}
    for index, exposure in enumerate((-2.0, -1.0, -0.5, 0.0, 0.5, 1.0)):
        noisy = np.clip(
            linear * 2**exposure + generator.normal(0, 0.01, linear.shape), 0, 1
        )
        path = directory / f"photo_{index}.png"
        Image.fromarray((noisy * 255).astype(np.uint8)).save(path)
        manifest["images"].append(
            {
                "name": path.name,
                "width": width,
                "height": height,
                "exposure_stops": exposure,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "examples/assets/sa1",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    if not 16 <= args.width <= 7680 or not 16 <= args.height <= 4320:
        parser.error("Fixture dimensions out of range")
    print(json.dumps(prepare(args.output, args.width, args.height), indent=2))


if __name__ == "__main__":
    main()
