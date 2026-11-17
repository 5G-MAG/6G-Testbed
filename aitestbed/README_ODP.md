# One-D-Piece (ODP) <-> Qwen3-VL MMBench, lossy UDP link

An end-to-end lossy-link image experiment built on top of `aicodec`/`aitestbed`:
One-D-Piece L-256 tokenizes MMBench images, sends them over UDP under a
network profile, the cloud server reconstructs the survivors, and Qwen3-VL
scores the reconstructed folder offline.

> All paths and commands below are relative to the repo root.

**Features:**
- **Whole-image UDP transport** — ODP encode -> 12-bit token packing ->
  UDP datagrams -> decode + PNG reconstruction (`chunked` mode not usable
  yet, see "Known limitations")
- **Two independent loss mechanisms** — real `tc/netem` link loss
  (`--profiles lossy`) and payload-level synthetic token drop at the
  client (`--profiles no_emulation` + `--payload-drop-mode`)
- **One-click pipeline** (`run_odp_mmbench_pipeline.py`) — server start ->
  orchestrator scenario -> reconstruction check -> Qwen3-VL scoring ->
  per-profile + combined JSON summaries
- **Resumable dataset generation** (`tsv2figure.py`) — official MMBench
  TSV -> jpgs + `manifest.jsonl`, safe to re-run on a partial output dir

## Weights and dataset

Not bundled with this repo — download separately and place at:

| Path                                                                          | Contents               | Size    | Download                                                                                            |
| ----------------------------------------------------------------------------- | ---------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `aicodec/aicodec/models/one_d_piece/checkpoints/pytorch_model.bin`            | ODP L-256 tokenizer    | ~2.4 GB | [turing-motors/One-D-Piece-L-256](https://huggingface.co/turing-motors/One-D-Piece-L-256/tree/main) |
| `aicodec/aicodec/models/Qwen3_VL_30B_A3B_Instruct/` | Qwen3-VL scoring model | ~60 GB  | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)   |

`aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN/` ships empty: download the
official MMBench TSV and generate the images + manifest locally:

```bash
pip install -r requirements_qwen.txt   # isolated env for the eval step
python aicodec/aicodec/dataset/tsv2figure.py \
    --tsv <path-to-mmbench_dev_en.tsv> \
    --out-dir aicodec/aicodec/dataset/MMBench/MMBench_DEV_EN
```

## Quick Start

```bash
# smoke test with a handful of images before a full run
python run_odp_mmbench_pipeline.py \
    --mode whole --profiles lossy --max-images 20 --eval-limit 20

# full run: one network profile
python run_odp_mmbench_pipeline.py \
    --mode whole --profiles lossy --interface lo \
    --eval-conda-env qwen3vl --eval-gpu 1

# full run: sweep several network profiles in one go
python run_odp_mmbench_pipeline.py \
    --mode whole --profiles no_emulation lossy 6g_itu_hrllc cell_edge --interface lo

# payload-level synthetic token drop (whole-mode only), independent of netem
python run_odp_mmbench_pipeline.py --mode whole --profiles no_emulation \
    --payload-drop-mode tail --payload-drop-pct 30
```

## Key CLI Flags

| Flag                                         | Default                        | Meaning                                                                           |
| -------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------- |
| `--mode`                                     | (none)                         | `whole`/`chunked` (chunked currently non-functional)                              |
| `--profiles`                                 | `["lossy"]`                    | one or more `aitestbed/configs/profiles.yaml` profiles; each gets an isolated run |
| `--interface`                                | `lo`                           | NIC netem is applied to                                                           |
| `--max-images` / `--eval-limit`              | unlimited                      | cap sent / scored images (smoke tests)                                            |
| `--server-port`                              | `1237`                         | ODP UDP server listen port                                                        |
| `--out-dir`                                  | `results/odp_mmbench_pipeline` | output root                                                                       |
| `--skip-eval`                                | off                            | link/reconstruction only, skip Qwen3-VL scoring                                   |
| `--eval-conda-env`                           | `qwen3vl`                      | conda env `mmbench_eval.py` runs in                                               |
| `--payload-drop-mode` / `--payload-drop-pct` | off                            | synthetic token drop; whole-mode only, only active under `no_emulation`           |

## Network Profiles (ODP-specific)

In addition to the shared profiles in the root [Network Profiles](./README.md#network-profiles)
table, `no_emulation` (0% netem loss/delay, see `aitestbed/configs/profiles.yaml`)
doubles as the carrier profile for `--payload-drop-mode`/`--payload-drop-pct`
(app-layer, exact ratio): `odp_image_udp.py` only applies the drop when
`network_profile == "no_emulation"`, so it never stacks with real netem loss
on `lossy`/`5g_urban`/etc. `payload_drop_mode`/`payload_drop_pct` default to
`none`/`0`, so a plain `no_emulation` baseline run (including as the
zero-loss control point in a multi-profile sweep) stays drop-free unless you
pass `--payload-drop-mode`/`--payload-drop-pct` explicitly.

## Output layout

```
<out-dir>/<profile_tag>/
    recon/                  # reconstructed images + eval manifest.jsonl
    eval/                   # Qwen3-VL summary.json + predictions
    reports/                # server-side link report
    logs/                   # server/orchestrator/eval logs
    pipeline_summary.json
<out-dir>/combined_summary.json   # one row per profile, for spreadsheet comparison
```

Key fields: `link_report.token_delivery_rate`, `eval.raw_accuracy` /
`filtered_accuracy` / `invalid_predictions`.
