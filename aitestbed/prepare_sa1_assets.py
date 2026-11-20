#!/usr/bin/env python3
"""Download six public HDR+ raw frames and prepare documented 4K derivatives."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

SOURCE = "https://hdrplusdata.org/dataset.html"
LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"
BURST = "0006_20160722_115157_431"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output, burst=BURST):
    import rawpy

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["burst"] != burst:
            raise ValueError(
                "Existing assets belong to a different burst; choose a fresh output directory"
            )
        for item in manifest["files"]:
            if digest(output / item["path"]) != item["sha256"]:
                raise ValueError("Asset changed: " + item["path"])
        return manifest
    prefix = f"20171106_subset/bursts/{burst}/"
    response = httpx.get(
        "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o",
        params={"prefix": prefix, "maxResults": 100},
        timeout=30,
    )
    response.raise_for_status()
    objects = sorted(
        (v for v in response.json()["items"] if v["name"].endswith(".dng")),
        key=lambda v: v["name"],
    )[:6]
    if len(objects) != 6:
        raise ValueError("Six raw frames are required")
    manifest = {
        "dataset": "HDR+ Burst Photography Dataset",
        "source": SOURCE,
        "attribution": "Hasinoff, Sharlet, Geiss, Adams, Barron, Kainz, Chen and Levoy (2016)",
        "license": LICENSE,
        "burst": burst,
        "workload_kind": "public_raw_burst_4k_crop",
        "caveat": "Same-exposure mobile burst, not exposure bracketing. 4K crops and rendered JPEG proxies are derivatives; actual compressed bytes are measured, not forced to 20 MB.",
        "files": [],
        "photo_paths": [],
        "raw_frames": [],
    }

    def fetch(item):
        url = "https://storage.googleapis.com/hdrplusdata/" + item["name"]
        path = output / Path(item["name"]).name
        if not path.exists() or path.stat().st_size != int(item["size"]):
            with httpx.stream("GET", url, timeout=90) as response:
                response.raise_for_status()
                temporary = path.with_suffix(".partial")
                with temporary.open("wb") as f:
                    for data in response.iter_bytes():
                        f.write(data)
                if temporary.stat().st_size != int(item["size"]):
                    raise ValueError("Incomplete download")
                temporary.replace(path)
        return path, url

    with ThreadPoolExecutor(max_workers=3) as executor:
        downloads = list(executor.map(fetch, objects))
    planes = {}
    for index, (path, url) in enumerate(downloads):
        manifest["files"].append(
            {
                "path": path.name,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
        with rawpy.imread(str(path)) as raw:
            plane = raw.raw_image_visible
            h, w = plane.shape
            if h < 2160 or w < 3840:
                raise ValueError(f"Raw frame too small: {w}x{h}")
            y, x = ((h - 2160) // 2) // 2 * 2, ((w - 3840) // 2) // 2 * 2
            planes[f"frame_{index}"] = plane[y : y + 2160, x : x + 3840].astype("<u2")
            # Crop in sensor coordinates before applying camera orientation.
            # LibRaw's automatic rotation otherwise swaps the RGB dimensions
            # and Pillow silently pads an out-of-bounds landscape crop black.
            camera_flip = raw.sizes.flip  # LibRaw mutates this during postprocess.
            rgb = raw.postprocess(
                use_camera_wb=True, no_auto_bright=True, output_bps=8, user_flip=0
            )
            if rgb.shape[:2] != plane.shape:
                raise ValueError("Rendered and sensor coordinates differ")
            rendered = Image.fromarray(rgb).crop((x, y, x + 3840, y + 2160))
            rotation = {
                3: Image.Transpose.ROTATE_180,
                5: Image.Transpose.ROTATE_90,
                6: Image.Transpose.ROTATE_270,
            }.get(camera_flip)
            if rotation is not None:
                rendered = rendered.transpose(rotation)
            photo = output / f"photo_{index}.jpg"
            rendered.save(photo, quality=95, subsampling=0)
            manifest["photo_paths"].append(str(photo.resolve()))
            manifest["files"].append(
                {
                    "path": photo.name,
                    "bytes": photo.stat().st_size,
                    "sha256": digest(photo),
                    "transformation": "camera white balance, demosaic without rotation, 8-bit sRGB, even-origin sensor crop 3840x2160, camera orientation, JPEG quality 95, 4:4:4",
                    "width": rendered.width,
                    "height": rendered.height,
                    "camera_flip": camera_flip,
                }
            )
            manifest["raw_frames"].append(
                {
                    "source": path.name,
                    "source_shape": [h, w],
                    "crop_xy": [x, y],
                    "shape": [2160, 3840],
                    "raw_pattern": raw.raw_pattern.tolist(),
                    "black_level_per_channel": raw.black_level_per_channel,
                    "white_level": raw.white_level,
                }
            )
    raw_path = output / "six_frames_4k_raw.npz"
    np.savez_compressed(raw_path, **planes)
    manifest["raw_payload"] = str(raw_path.resolve())
    manifest["files"].append(
        {
            "path": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "sha256": digest(raw_path),
            "transformation": "Six uint16 Bayer crops in NPZ/DEFLATE; metadata in manifest",
        }
    )
    manifest["raw_uncompressed_bytes"] = sum(v.nbytes for v in planes.values())
    manifest["raw_compressed_bytes"] = raw_path.stat().st_size
    manifest["proxy_jpeg_bytes"] = sum(
        Path(p).stat().st_size for p in manifest["photo_paths"]
    )
    thumb = output / "image_condition.jpg"
    image = Image.open(manifest["photo_paths"][0])
    image.thumbnail((512, 512))
    image.save(thumb, quality=90)
    manifest["image_condition"] = str(thumb.resolve())
    manifest["files"].append(
        {
            "path": thumb.name,
            "bytes": thumb.stat().st_size,
            "sha256": digest(thumb),
            "transformation": "512-pixel thumbnail of first rendered crop, JPEG quality 90",
        }
    )
    (output / "ATTRIBUTION.md").write_text(
        f"# HDR+ evaluation sample\n\n{manifest['attribution']}\n\nSource: {SOURCE}\n\nLicense: {LICENSE}\n\nBurst: {burst}. Derivatives retain CC BY-SA 4.0; conversion steps and hashes are in manifest.json.\n"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--burst", default=BURST)
    args = parser.parse_args()
    m = prepare(args.output, args.burst)
    print(
        json.dumps(
            {k: m[k] for k in ["burst", "raw_compressed_bytes", "proxy_jpeg_bytes"]},
            indent=2,
        )
    )
