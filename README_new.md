# 6G AI Traffic Characterization Testbed

A framework for measuring and analyzing AI/LLM service traffic patterns under various network conditions, designed to support 3GPP SA4 6G Media Study contributions.

## Components

| Component | Description |
|-----------|-------------|
| [aitestbed/](./aitestbed/) | Main testing framework for running AI traffic experiments |
| [netemu/](./netemu/) | Network emulation library wrapping Linux tc/netem |
| [aiortc-main-clean/](./aiortc-main-clean/) | aiortc with VLM tokenizer and LLM processing for token transmission over RTP |

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
- VLM_tokenizer: download vqgan.ckpt and vqgan.yaml, put them in aiortc-main-clean/src/aiortc/liquid/checkpoints/chameleon  
https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt  
https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml  
- VLM model: download all model files, put them in aiortc-main-clean/src/aiortc/liquid/checkpoints/model  
https://huggingface.co/Junfeng5/Liquid_V1_7B/tree/main  
- dataset (proactive output): download dataset, unzip to aitestbed/examples/assets/dataset  
https://huggingface.co/datasets/mjuicem/StreamingBench/blob/main/Proactive%20Output_1-25.zip
*** scenario config ***
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

- another dataset (real): download dataset, unzip to aitestbed/examples/assets/dataset_real
https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/main/Real-Time%20Visual%20Understanding_1-50.zip
*** scenario config ***
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

**Preparation for Chat with token ID scenario:**
- Set up OpenAI-compatible server with open-source LLM
- Copy tokenizer related files to TOK_PATH
- Configure .env file with
  - OPENAI_BASE_URL
  - OPENAI_API_KEY
  - MODEL_NAME
  - TOK_PATH


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
pip install -e aiortc-main-clean
pip install --use-deprecated=legacy-resolver -r aitestbed/requirements.txt # do not contain aiortc

# or you can use uv
uv venv
source .venv/bin/activate
uv pip install -e netemu
uv pip install -e aiortc-main-clean
uv pip install -r aitestbed/requirements.txt # do not contain aiortc

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
