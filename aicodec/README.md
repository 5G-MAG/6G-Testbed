# aicodec

AI codec plugin for [aiortc](https://github.com/aiortc/aiortc): token-based
semantic communication codecs for VLM streaming.

Registers three custom video codecs (`video/tokenId`, `video/tokenIdTransparent`,
`video/tokenIdLLM`) plus an RTX variant into a stock `aiortc` install at
runtime — no forked aiortc needed. Wire format is model-agnostic; the VLM
backend is pluggable.

## Layout

```
aicodec/
├── aicodec/
│   ├── __init__.py    # register() / set_model() + re-exports
│   ├── register.py    # aiortc monkeypatch core (5 patches)
│   ├── shared.py      # loop / inference queue / selected model name
│   ├── codec.py       # codec entry (re-exports from tokenid.py)
│   ├── tokenid.py     # token codec implementation (encoder/decoder/depayload)
│   └── models/        # pluggable VLM model layer
│       ├── base.py    # VLMBridge abstract base
│       └── liquid/    # first model: bridge.py + vendored source (src/) + checkpoints/
```

## Install

```bash
pip install aiortc>=1.14.0,<1.16.0   # stock dependency (1.15 recommended)
pip install -e "aicodec" --config-settings editable_mode=compat
```

> **Note (editable installs):** the `--config-settings editable_mode=compat`
> flag is required when running `pip install -e aicodec` from a parent
> directory that already contains a plain `aicodec/` folder (such as this
> repo): the default modern editable finder lets that directory shadow the
> installed package when Python's working directory is the repo root. The
> compat mode installs a `.pth` path entry instead, which resolves correctly.

Optional extras: `pip install -e "aicodec[liquid]"` pulls in torch / transformers.

## Usage

```python
import aicodec
aicodec.register()    # call once before creating an RTCPeerConnection
aicodec.set_model("liquid")  # optional: select the VLM model (default "liquid")

from aicodec.codec import TokenIdEncoder, AsyncTokenIdDecoder
import aicodec.shared as shared   # shared.loop etc.
```

## Pluggable VLM models

Model backends live in `aicodec/models/` and are selected by name:

```python
from aicodec import available_models
available_models()      # ['liquid', ...]
```

The aitestbed realtime scenarios negotiate the model with the server over TCP
signaling (`{"type": "model", "model": <name>}` → `{"type": "model_ack", ...}`)
before the SDP exchange, driven by `model:` in `configs/scenarios.yaml`.

### Adding a new model

1. Create `aicodec/aicodec/models/<name>/`:
   - `bridge.py` — subclass `aicodec.models.VLMBridge`, implement
     `setup` / `tokenize` / `detokenize` / `generate` / `estimate_token_number`
   - `__init__.py` — `register_model("<name>", <BridgeCls>)`
2. Vendor third-party code under `models/<name>/src/` and put checkpoints
   under `models/<name>/checkpoints/`
3. Select it from the testbed config (`model: <name>` in scenarios.yaml)

The RTP wire format is model-agnostic — only the bridge layer changes.

## Checkpoints (liquid)

Download `vqgan.ckpt` / `vqgan.yaml` and the LLM model files, and place them at:

```
aicodec/aicodec/models/liquid/checkpoints/chameleon/   # vqgan.yaml, vqgan.ckpt
aicodec/aicodec/models/liquid/checkpoints/model/       # LLM files
```

For an editable install you may also symlink these directories.

## Upstream license notice

`models/liquid/src/` is derived from the [Liquid](https://github.com/FoundationVision/Liquid)
project (MIT). Chameleon inference code originates from
[Chameleon](https://github.com/facebookresearch/chameleon) (BSD-3-Clause) via Liquid.
See `LICENSE` files under `models/liquid/src/`.
