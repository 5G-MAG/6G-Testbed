# 6G AI Traffic Characterization Testbed

A framework for measuring and analyzing AI/LLM service traffic patterns under various network conditions, designed to support 3GPP SA4 6G Media Study contributions.

## Components

| Component | Description |
|-----------|-------------|
| [aitestbed/](./aitestbed/) | Main testing framework for running AI traffic experiments |
| [netemu/](./netemu/) | Network emulation library wrapping Linux tc/netem |
| [aicodec/](./aicodec/) | AI codec plugin for aiortc (token semantic communication: VLM tokenizer and LLM processing over RTP) |

## aitestbed

The core testing framework that orchestrates experiments across multiple AI providers and scenarios.

**Features:**
- **11 scenario types**: Chat, agentic AI with MCP tools, image generation, multimodal, video understanding, realtime WebSocket/WebRTC
- **8 LLM providers**: OpenAI, Gemini, DeepSeek, vLLM, and realtime variants
- **60+ metrics**: TTFT/TTLT, latency percentiles, UL/DL ratios, token rates, agent loop factors
- **Multi-layer traffic capture**: L3/L4 via tcpdump, L7 via mitmproxy
- **SQLite logging** with structured metrics schema

## New scenarios
Realtime video analysis and chat with token ID scenarios are supported.

**Preparation for realtime video analysis scenario:**
- VLM_tokenizer:
  - Download  [vqgan.ckpt](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt) and [vqgan.yaml](https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml) and store them in `aicodec/aicodec/models/liquid/checkpoints/chameleon`
- VLM model:
  - Download all these [model files](https://huggingface.co/Junfeng5/Liquid_V1_7B/tree/main) and store them in `aicodec/aicodec/models/liquid/checkpoints/model`  

- Dataset (proactive output):
  - Download [this dataset](https://huggingface.co/datasets/mjuicem/StreamingBench/blob/main/Proactive%20Output_1-25.zip) and unzip to `aitestbed/examples/assets/dataset`  

  ***Scenario Configuration***
  ```
  realtime_video_understanding:
    type: "realtime_video_understanding"
    description: "Real-time visual understanding via VLM model"
    provider: "vlm-local"
    model: "liquid"
    target_fps: 2
    stream: true
    task_config:
      task_class: "proactive_output"
      po_thd: 3 
      results_dir: "examples/assets/dataset/"
      video_paths: "examples/assets/dataset/"
    prompts:
      - "What is in the video? Answer directly!"
  ```
- Dataset (real):
  - Download this [dataset](https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/main/Real-Time%20Visual%20Understanding_1-50.zip) and unzip to `aitestbed/examples/assets/dataset_real`

  ***Scenario Configuration***
  ```
  realtime_video_understanding:
    type: "realtime_video_understanding"
    description: "Real-time visual understanding via VLM model"
    provider: "vlm-local"
    model: "liquid"
    target_fps: 2
    stream: true
    task_config:
      task_class: "real"
      po_thd: 3 # not used
      results_dir: "examples/assets/dataset_real/"
      video_paths: "examples/assets/dataset_real/"
    prompts:
      - "What is in the video? Answer directly!"
  ```
**Preparation for Chat with token ID scenario:**
- Set up OpenAI-compatible server with open-source LLM
- Copy tokenizer related files to TOK_PATH
- Configure .env file with
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - MODEL_NAME
  - TOK_PATH

## Encoder Placement (visual encoder down-placement)

A dedicated experiment family that measures the traffic/QoE impact of moving the **vision encoder (ViT) of a VLM from the server into the terminal** for video understanding:

- **`server` mode** — the terminal uploads the original video; the server-side SGLang engine runs decode + processor + ViT + prefill + decode.
- **`edge` mode** — the terminal runs decode + processor + ViT locally and uploads a serialized `precomputed_embedding` bundle (MBs) instead of the raw mp4 (hundreds of MBs); the server only performs prefill + decode on the precomputed latent features.
- **Zero-training compression stack** (edge mode): VidCom2 dynamic video-token pruning, optional fp8 (e4m3, per-tensor scale) feature quantization, and ratio-based frame sampling — up to ~49x upload-size reduction on the reference clip.
- **Concurrent arms**: barrier-aligned waves of N terminal users (thread or process topology, symmetric CPU budgets) against a lock-free concurrent sidecar.

Two parallel model families (standalone-sibling file sets, zero cross-dependencies):

| Family | Model | Serial sidecar | Concurrent sidecar | Serial runner | Concurrent runner |
|--------|-------|---------------|--------------------|---------------|-------------------|
| Qwen3-VL | `Qwen3-VL-30B-A3B-Instruct` (also 4B) | port 8002 | port 8012 | yaml/env entry (`configs/scenarios.yaml`, scenario `encoder_placement`) — no dedicated runner script | `run_encoder_placement_concurrent.py` |
| Qwen3.6 | `Qwen3.6-35B-A3B(-FP8)` | port 8004 | port 8014 | `run_encoder_placement_qwen36.py` | `run_encoder_placement_concurrent_qwen36.py` |

### Two-tier retention split (VidCom2 layered transfer)

With `compressor: vidcom2` in edge mode, `optional_ratio` (0 <= o < retention_ratio) splits the VidCom2-retained token set into two waves:

- **Must tier** (`retention_ratio - optional_ratio`, the most-critical tokens) rides the reliable first wave: the serialized bundle `PUT` over TCP. The must bundle alone is a complete, prefillable `precomputed_embedding` — identical to a single-tier run at the reduced base ratio, so accuracy degrades gracefully toward that floor as optional tokens are lost.
- **Optional tier** (the least-critical tail of the retained set) is streamed as best-effort **E2U datagrams over UDP** between the must `PUT` and the chat `POST`. Each part is self-describing (row ids, slot positions, mrope columns); the server merges tokens back at their per-token insertion positions. Parts (or fragments) that miss the server's merge snapshot at chat-arrival are counted as `late_dropped` and **never retransmitted** — loss genuinely reaches the application layer.

**E2U wire format**: 30-byte header (magic `E2U1`, bundle id, part seq, fragment index/count, payload length) + up to 1370 B of payload per datagram. A part larger than one datagram is fragmented; if any fragment is lost the whole part is dropped (true UDP semantics — no retransmission, no CRC).

**Prerequisite**: the sidecar must be started with `--opt-udp-port > 0` (serial default 8005, concurrent default 8015; `start_sidecar*.sh` forwards extra args). The orchestrator pre-checks the receiver when `optional_ratio > 0`, shapes the UDP port pair on loopback together with the TCP port under netem profiles, and extends the `--capture-pcap` filter automatically.

**Knobs** (bench drivers and scenario config; env fallbacks `ENCPL_OPTIONAL_RATIO` / `ENCPL_OPT_TOKENS_PER_PACKET` / `ENCPL_OPT_SEND_GAP_MS` / `ENCPL_OPT_CHAT_DELAY_MS` win over yaml keys):

| Knob | Default | Meaning |
|------|---------|---------|
| `optional_ratio` / `--optional-ratio` | 0 (off) | size of the best-effort tier; 0 = single-tier legacy path |
| `--opt-tokens-per-packet` | 64 | packaging granularity only — the server merges per-token, so this does not affect accuracy |
| `--opt-send-gap-ms` | 0.05 | sleep between adjacent UDP datagrams (paces the wave below the receiver's reassembly rate; 0 = blast, needs a raised SO_RCVBUF) |
| `--opt-chat-delay-ms` | 0 | sleep after the UDP wave and before the chat POST — trades optional-token arrival rate against TTFT |

**Minimal example** (serial Qwen3.6 arm):

```bash
# sidecar with the optional-tier receiver on
bash start_sidecar_qwen36.sh --opt-udp-port 8005

# bench: 30% retained, of which the last 10% goes best-effort
python tools/encpl_bench_eval_qwen36.py \
    --dataset mlvu --dataset-dir ./MLVU \
    --mode edge --compressor vidcom2 \
    --retention-ratio 0.3 --optional-ratio 0.10
```

For the two-tier sweep over absolute optional-retention values (`-opt{g}` cell naming), see `tools/run_qwen36_matrix.py` and its `--optional-retention` matrix below.

### Qwen3.6 — serial benchmark (tested)

```bash
# 1) start the sidecar (bootstraps conda env, CUDA_HOME, sglang patches)
bash start_sidecar_qwen36.sh \
    --model aicodec/aicodec/models/Qwen3.6-35B-A3B \
    --mem-fraction-static 0.85

# 2) run the benchmark (from aitestbed/)
python tools/encpl_bench_eval_qwen36.py \
    --dataset mlvu --dataset-dir ./MLVU \
    --base-url http://localhost:8004 \
    --limit 10 --mode edge --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --edge-cpus "1-65" \
    --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B
    
python tools/encpl_bench_eval_qwen36.py \
     --dataset mlvu --dataset-dir ./MLVU \
     --base-url http://localhost:8004 
     --limit 10 --mode server  
     --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B
```

| Parameter               | Meaning                                                                                      |
|-------------------------|----------------------------------------------------------------------------------------------|
| `--model`               | Weights path. Sidecar: engine weights (bf16 35B MoE here; FP8 variant is the built-in default). Bench tool: terminal-side ViT weights in edge mode (server mode ignores it) |
| `--mem-fraction-static` | GPU KV-cache fraction reserved by SGLang (0.85 on an 80GB card)                              |
| `--dataset`             | Benchmark set: `mlvu` or `videomme`                                                          |
| `--dataset-dir`         | Local dataset root (videos + question json/parquet)                                          |
| `--base-url`            | Sidecar endpoint (serial Qwen3.6 sidecar = port 8004)                                        |
| `--limit 10`            | Cap the number of questions evaluated                                                        |
| `--mode edge`           | Terminal runs decode + processor + ViT locally, uploads the precomputed-embedding bundle     |
| `--mode server`         | Terminal uploads the original video; the sidecar runs the full pipeline (baseline arm)       |
| `--vision-only`         | Load only the vision tower on the terminal (~1 GB VRAM instead of ~35 GB)                    |
| `--compressor vidcom2`  | VidCom2 dynamic video-token pruning of the uploaded bundle                                   |
| `--retention-ratio 0.3` | Keep 30% of video tokens after pruning                                                       |
| `--bundle-quant fp8`    | Quantize bundle features to fp8 e4m3 (halves feature bytes; requires `--compressor vidcom2`) |
| `--edge-cpus 1-65`      | Pin the terminal-side work to cores 1-65 (disjoint from the sidecar's cores)                 |

**Stream safety timeouts** — available on every encoder-placement bench/runner (both model families). A wedged sidecar stream used to stall a run forever (observed: a 29.5 h frozen matrix); these caps turn that into a per-request exception:

| Parameter | Default (serial / concurrent) | Meaning |
|-----------|-------------------------------|---------|
| `--stream-connect-timeout` | 20 s | TCP connect timeout for sidecar HTTP requests (loopback handshake with netem'd SYN retransmits) |
| `--stream-first-event-timeout` | 200 s / 900 s | Budget from chat POST to the first SSE event — covers server-side edge prepare + queueing + prefill + TTFT. Concurrent tools default 900: barrier waves of N users against a small running-request cap queue tail users behind minutes of prefill |
| `--stream-idle-timeout` | 20 s | Max silent gap between SSE events mid-stream (tightened onto the socket right after the first event) |
| `--stream-upload-read-timeout` | 1200 s | Read timeout for the bundle-upload PUT response (covers the server's store+ack after the possibly minutes-long upload-direction transfer) |

`0` disables any guard (restores the historical unlimited wait). Equivalent env overrides for ad-hoc runs: `ENCPL_STREAM_CONNECT_TIMEOUT`, `ENCPL_STREAM_FIRST_EVENT_TIMEOUT`, `ENCPL_STREAM_IDLE_TIMEOUT`, `ENCPL_STREAM_UPLOAD_READ_TIMEOUT` (explicit CLI flags win).

### Qwen3.6 — concurrent benchmark (tested)

```bash
# 1) start the concurrent sidecar
bash start_sidecar_concurrent_qwen36.sh 
    --max-running-requests 2 --mm-processor-worker-num 4 \
    --decode-threads 2 --preprocess-threads 4 \
    --model aicodec/aicodec/models/Qwen3.6-35B-A3B \
    --server-cpus "0-31" --enable-mixed-chunk \
    --mm-io-worker-num 4 --mem-fraction-static 0.85 \
 

# 2) run the concurrent benchmark (from aitestbed/)
python tools/encpl_bench_eval_concurrent_qwen36.py \
	--mode server --users 8 \
    --base-url http://localhost:8014 \
    --runs 1 --db logs/encpl_conc_server_n4.db \
    --edge-cpu-ids "32-58" --edge-cpus-per-worker 3 \
    --enable-multi-processes --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU --limit 32

python tools/encpl_bench_eval_concurrent_qwen36.py \
	--mode edge --users 8 --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --base-url http://localhost:8014 \
    --runs 1 --db logs/encpl_conc_edge_n4.db \
    --model ../aicodec/aicodec/models/Qwen3.6-35B-A3B \
	--enable-multi-processes --edge-cpu-ids "32-58" \
	--edge-cpus-per-worker 3 --decode-threads 2 \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU	--limit 32
```

Same parameters as above, plus:

**Sidecar (concurrent) parameters:**

| Parameter                     | Meaning                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `--max-running-requests 2`    | Scheduler cap on requests batched on the GPU at once; above it, requests queue                   |
| `--mm-io-worker-num 4`        | Width of the server's video I/O thread pool (decode + sample + resize); at most 4 videos decode in parallel |
| `--mm-processor-worker-num 4` | Width of the multimodal processor pool (HF processor calls: normalize / tensorize)               |
| `--decode-threads 2`          | Cap the ENGINE-side decode thread pool at 2 (restart to change)                                  |
| `--preprocess-threads 4`      | Cap the ENGINE-side CPU preprocessing (resize/processor) threads at 4                            |
| `--server-cpus 0-31`          | Pin the sidecar and its scheduler subprocess to cores 0-31 (disjoint from the terminals' pool)   |
| `--enable-mixed-chunk`        | Mix decode requests into prefill chunks so in-flight decodes don't stall behind long prefills   |

**Bench (concurrent) parameters:**

| Parameter                  | Meaning                                                                            |
|----------------------------|------------------------------------------------------------------------------------|
| `--users 8`                | 8 barrier-aligned concurrent terminal users per wave                               |
| `--enable-multi-processes` | One OS process per user (own ViT / GIL / connection — truly independent terminals) |
| `--edge-cpu-ids 32-58`     | CPU pool the workers are pinned to, round-robin                                    |
| `--edge-cpus-per-worker 3` | CPUs each worker gets from the pool                                                |

### Qwen3-VL — serial benchmark (tested)

```bash
# 1) start the sidecar
bash start_sidecar.sh \
    --model aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
    --mem-fraction-static 0.85

# 2) run the benchmark (from aitestbed/)
python tools/encpl_bench_eval.py \
    --dataset mlvu --dataset-dir ./MLVU \
    --base-url http://localhost:8002 \
    --limit 10 --mode edge --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8
    
python tools/encpl_bench_eval.py \ 
    --dataset mlvu --dataset-dir ./MLVU \
    --mode server --limit 10 \
    --base-url http://localhost:8002 \ 
    --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct/
```

Same parameter semantics as the Qwen3.6 serial run; the un-suffixed tool talks to the Qwen3-VL sidecar (port 8002, `start_sidecar.sh`) and defaults its edge model to the 4B weights.

### Qwen3-VL — concurrent benchmark (tested)

```bash
# 1) start the concurrent sidecar
bash start_sidecar_concurrent.sh \
    --max-running-requests 2 \
    --model ./aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct/ \
    --server-cpus "0-31" --enable-mixed-chunk --mm-io-worker-num 4 \
    --mm-processor-worker-num 4  --decode-threads 2 --preprocess-threads 4

# 2) run the concurrent benchmark (from aitestbed/)
python tools/encpl_bench_eval_concurrent.py \
	--mode edge --users 8 --vision-only \
    --compressor vidcom2 --retention-ratio 0.3 --bundle-quant fp8 \
    --base-url http://localhost:8012 \
    --runs 1 --db logs/encpl_conc_edge_n4.db --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
	--enable-multi-processes --edge-cpu-ids "1-65" --edge-cpus-per-worker 3 --decode-threads 2 \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU	--limit 32	
    
python tools/encpl_bench_eval_concurrent.py \
	--mode server --users 8 \
    --base-url http://localhost:8012 \
    --runs 1 --db logs/encpl_conc_server_n4.db \
    --edge-cpu-ids "1-65" --edge-cpus-per-worker 3 \
    --enable-multi-processes --model ../aicodec/aicodec/models/Qwen3-VL-30B-A3B-Instruct \
	--dataset mlvu --dataset-dir /data01/mlvu/MLVU --limit 32
```

| Parameter (runner-only) | Meaning |
|------------------------|---------|
| `--runs N` | Number of repetitions (waves x prompts per run) recorded to the DB |
| `--db PATH` | Output SQLite traffic DB (use a separate one per arm for A/B comparison) |
| `--decode-threads 2` | Cap the terminal video-decode thread pool at 2 (decord/torchcodec otherwise size it by host core count) |

The Qwen3-VL sidecar/bench flags carry the same meanings as the Qwen3.6 tables above (`--max-running-requests`, `--server-cpus`, `--enable-mixed-chunk`, `--mm-io-worker-num`, `--mm-processor-worker-num`, `--decode-threads`, `--preprocess-threads`, `--users`, `--enable-multi-processes`, `--edge-cpu-ids`, `--edge-cpus-per-worker`).

### Qwen3.6 — full-matrix driver (`tools/run_qwen36_matrix.py`)

Drives the whole Qwen3.6 experiment unattended: serial + concurrent phases, server/edge arms, retention/quant ladders, all network profiles, per-cell pcap + SQLite, and a final aggregated report. Sidecars are started/stopped automatically (auto mode).

```bash
# plan only — render every cell's exact command/netem/filter
python aitestbed/tools/run_qwen36_matrix.py --dry-run

# full run (defaults: serial 50 videos/cell, concurrent 3 videos/user,
# profiles no_emulation,5g_urban,wifi_good,cell_edge)
python aitestbed/tools/run_qwen36_matrix.py \
    --server-gpu 0 --client-gpu 1 \
    --sidecar-mem-frac 0.92 --conc-mem-frac 0.85

# resume after an interruption (completed cells are skipped, leftover
# artifacts of unfinished cells are cleaned automatically)
python aitestbed/tools/run_qwen36_matrix.py \
    --resume logs/q36m/manifest_<ts>.json

# regenerate the report from an existing manifest
python aitestbed/tools/run_qwen36_matrix.py --report-only logs/q36m/manifest_<ts>.json
```

**Matrix scale knobs:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--serial-limit` | 50 | Videos per serial cell |
| `--conc-videos-per-user` | 3 | Videos per concurrent user; cell limit = V × users |
| `--phases` | both | `serial` / `concurrent` / `both` |
| `--profiles` | no_emulation,5g_urban,wifi_good,cell_edge | Network profiles per cell |
| `--retentions` / `--quants` / `--users` | 0.8,0.7,0.6,0.4,0.3 / fp8,bf16 / 2,4,8,16 | Ladders of the matrix |
| `--optional-retention` | (empty) | Comma list of ABSOLUTE optional-retention (o) values for the two-tier ladder, e.g. `0.06,0.12,0.18` — every (r, o) pair becomes a tiered cell in BOTH phases (invalid pairs with `0 < o < r` are warned and skipped). Empty keeps the legacy derived value `o = r*0.30` per retention tier |

Two-tier cell naming (both phases share one convention — no explicit tier tag; the presence of `-opt{g}` IS the two-tier marker): serial `...-r0.6-fp8-opt0.18__<profile>` / single-tier `...-r0.6-fp8__<profile>`, concurrent `...-r0.6-fp8-opt0.18-n8__<profile>` / single-tier `...-r0.6-fp8-n8__<profile>`.

**Per-cell wall-clock caps** (P2 — a wedged runner used to freeze the whole matrix; now it is killed and the matrix moves on):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--serial-cell-timeout` | 36000 s (10 h) | Serial cells legitimately take hours under `cell_edge` (5 Mbit/s); 10 h leaves headroom |
| `--conc-cell-timeout` | 14400 s (4 h) | Concurrent cells measured in minutes; 4 h for the n16 tail |
| (either = 0) | — | Disables the cap (historical hang-forever behaviour) |

On timeout the driver kills the runner's whole process group (`killpg`), tears down netem + tcpdump, marks the cell `error: "cell timeout"` in the manifest, and continues with the next cell.

**Stream timeouts** — the five per-request knobs (`--stream-connect-timeout`, `--stream-first-event-timeout`, `--conc-stream-first-event-timeout`, `--stream-idle-timeout`, `--stream-upload-read-timeout`) are forwarded into every cell's runner; defaults 20 / 200 (serial) / 900 (concurrent) / 20 / 1200 s. The concurrent first-event budget is separate (`--conc-stream-first-event-timeout`) because barrier waves queue tail users behind minutes of prefill.

**Artifacts & preflight:** every cell writes `logs/q36m/<cell_id>.db`, a pcap under `aitestbed/results/captures/q36m/`, and a cell log under `logs/q36m/cell_logs/`. The preflight refuses to start when artifacts of a *would-be-fresh* run already exist — pass `--resume <manifest>` to skip completed cells and auto-clean leftovers of unfinished ones; for a from-scratch re-run delete (or archive) `logs/q36m/`, `logs/q36m/cell_logs/` and `aitestbed/results/captures/q36m/` first.

**GPU layout:** `--server-gpu` pins the sidecar engine's card (via `CUDA_VISIBLE_DEVICES`), `--client-gpu` pins every cell subprocess (terminal-side edge work). `--conc-mem-frac` (default 0.85) is the concurrent sidecar's `--mem-fraction-static` — 0.92 OOMs on a shared 80 GB card with CUDA-graph pools; keep 0.85 unless the card is exclusively the engine's.

Full documentation — sidecar flags, mode switching, compression knobs, CPU budgets, benchmark tools (Video-MME / MLVU), and the model-specific differences — is in the **Encoder Placement** section of [aitestbed/README.md](./aitestbed/README.md).

## netemu

A lightweight network emulation library providing a clean interface to Linux traffic control.

**Features:**
- Wraps `tc` and `netem` for delay, jitter, packet loss, and rate limiting
- Bidirectional shaping via IFB devices
- 27 predefined profiles including 3GPP 5QI mappings
- Context manager support for automatic cleanup

```python
from netemu import NetworkEmulator

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("5g_urban")  # 20ms delay, 0.1% loss, 100 Mbps
    # Run your tests here
# Rules automatically cleared
```

## Quick Start

```bash
# From the repo root:
# if you prefer pip to manage python package
python3 -m venv venv
source venv/bin/activate
pip install -e netemu
pip install "aiortc>=1.14.0,<1.16.0" # stock upstream aiortc
pip install -e aicodec --config-settings editable_mode=compat # AI codec plugin (dynamic registration)

# or you can use uv
uv venv
source .venv/bin/activate
uv pip install -e netemu
uv pip install "aiortc>=1.14.0,<1.16.0"
uv pip install -e aicodec
uv pip install -r aitestbed/requirements.txt

# for realtime video analysis scenario
cd aitestbed
# start server first
python ./server/realtime_vlm_server.py
# run test
python orchestrator.py --scenario realtime_video_understanding --profile 5g_urban --runs 10
# run test with trace saved
PROFILE=lossy
python orchestrator.py \
    --scenario realtime_video_understanding \
    --profile $PROFILE \
    --runs 20 \
    --interface lo \
    --capture-pcap \
    --capture-dir results/captures/vlm/${PROFILE} \
    --report results/reports/experiment_report_realtime_video_understanding_${PROFILE}.json \
    --db logs/traffic_logs_vlm_${PROFILE}.db \
    --egress-only

# for chat with token ID scenario
# start up OpenAI-compatible server (locally or remotely)
# edit .env file with above environment variables
# run test
python orchestrator.py --scenario chat_token --profile 5g_urban --runs 10
# run test with trace saved
PROFILE=lossy
python orchestrator.py \
    --scenario chat_token \
    --profile $PROFILE \
    --runs 10 \
    --interface eno1 \ # Replace 'eno1' with actual interface
    --capture-pcap \
    --capture-dir results/captures/${PROFILE} \
    --report results/reports/experiment_report_chat_token_${PROFILE}.json \
    --db logs/traffic_logs${PROFILE}.db \
```


## Network Profiles

| Profile | Delay | Loss | Rate | Use Case |
|---------|-------|------|------|----------|
| `ideal_6g` | 1ms | 0% | unlimited | Baseline |
| `5g_urban` | 20ms | 0.1% | 100 Mbps | Urban 5G |
| `wifi_good` | 30ms | 0.1% | 50 Mbps | Home WiFi |
| `cell_edge` | 120ms | 1% | 5 Mbps | Poor coverage |
| `satellite` | 600ms | 0.5% | 10 Mbps | LEO satellite |
| `lossy`     | 0ms | 10% | unlimited | High loss test |


## Requirements

- Python 3.10+
- Linux with `iproute2` (for network emulation)
- Sudo access or Docker with `NET_ADMIN` capability
- NVIDIA GPU with ~30GB VRAM (~2.5GB for VLM client, ~27GB for VLM server)
- CUDA Toolkit >= 12.1

For the encoder-placement experiments: a single 80GB-class GPU host (or two hosts — one for the terminal arm, one for the sidecar) and the `epd` conda environment — see [Environment Installation](#environment-installation).

## Environment Installation

The testbed uses two separate Python environments. Install the one that matches the scenarios you plan to run.

### 1. Base environment (orchestrator scenarios)

For the orchestrator-driven scenarios (`chat`, `chat_token`, `realtime_*`, `direct_web_search`, etc.):

```bash
python3 -m venv venv            # or: uv venv
source venv/bin/activate
pip install -e netemu
pip install "aiortc>=1.14.0,<1.16.0"                      # stock upstream aiortc
pip install -e aicodec --config-settings editable_mode=compat  # AI codec plugin
pip install -r aitestbed/requirements.txt
```

### 2. `epd` environment (encoder placement)

The frozen environment for the encoder-placement experiments (Qwen3-VL / Qwen3.6 sidecars + edge terminals): **Python 3.11, torch 2.13.0, transformers 5.12.1, sglang 0.5.20** — 234 pinned packages in [requirements-epd.txt](./requirements-epd.txt).

```bash
# 1) create the env
conda create -n epd python=3.11 -y
conda activate epd

# 2) local editable packages first
pip install -e aicodec --config-settings editable_mode=compat
pip install -e netemu

# 3) frozen dependency snapshot — MUST use --no-deps
pip install --no-deps -r requirements-epd.txt
```

Why `--no-deps`: the snapshot already pins every transitive dependency, and the source env ships with a known (harmless, pre-existing) protobuf 5.x vs grpcio/cutlass metadata conflict that pip's resolver would reject. Verify without installing: `pip install --dry-run --no-deps -r requirements-epd.txt`.

**CUDA toolkit (flashinfer JIT):** the sidecar's flashinfer kernels JIT-compile on first launch and need `nvcc` inside the env (the NVIDIA driver alone is not enough):

```bash
# align the version with torch.version.cuda of the installed torch
conda install -y -n epd -c nvidia cuda-nvcc cuda-cudart-dev
```

Alternatively, a pip-installed CUDA toolkit under `site-packages/nvidia/` is auto-discovered. The `start_sidecar*.sh` bootstrap scripts resolve CUDA_HOME automatically (external toolkit → conda env nvcc → pip cu13 prefix) — no manual exports needed.

**Model weights** — place under `aicodec/aicodec/models/` (git-ignored):

| Path | Used by |
|------|---------|
| `Qwen3.6-35B-A3B/` (or `-FP8/`) | Qwen3.6 serial/concurrent sidecar + edge terminal |
| `Qwen3-VL-30B-A3B-Instruct/` | Qwen3-VL sidecar (server arm) |
| `qwen3_vl_4B/` | Qwen3-VL edge terminal (default edge model) |

The VidCom2 `token_compressor` is vendored in-repo (`aicodec/aicodec/models/VidCom2/`, added to `sys.path` automatically) — no install needed.

**Benchmark datasets** — expected layout under `--dataset-dir`:

| Dataset | Layout |
|---------|--------|
| MLVU | `json/{1..9}_*.json` + `video/<task_dir>/*.mp4` (official release) |
| Video-MME | `data/test-00000-of-00001.parquet` + `videos/*.mp4` (HF official release) |

**Verification:** the first `start_sidecar*.sh` launch JIT-compiles flashinfer kernels (a few minutes, then cached). Afterward:

```bash
curl http://<host>:8004/stats      # serial Qwen3.6 sidecar
curl http://<host>:8014/stats      # concurrent Qwen3.6 sidecar
```

## Sudoers (for Network Emulation)
```bash
TCPDUMP_PATH=$(which tcpdump)
echo "$USER ALL=(ALL) NOPASSWD: /usr/sbin/tc, $TCPDUMP_PATH, /usr/sbin/modprobe, /usr/sbin/ip" \
  | sudo tee /etc/sudoers.d/testbed
sudo chmod 440 /etc/sudoers.d/testbed
```

## Related repos
* 6g ai testbed
https://github.com/5G-MAG/6G-Testbed

* aiortc
https://github.com/aiortc/aiortc/tree/main

* liquid model
https://github.com/FoundationVision/Liquid

* streaming bench
https://github.com/THUNLP-MT/StreamingBench/tree/main  
