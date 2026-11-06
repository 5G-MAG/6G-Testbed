"""Download the Liquid_V1_7B checkpoint into THIS directory.

Run from anywhere: the target is always the folder containing this
script (checkpoints/model/), which is exactly where bridge.py looks
for it. snapshot_download lays the repo out flat (config.json,
tokenizer files, model-0000X-of-00004.safetensors, index, ...) — the
same layout AutoTokenizer/AutoModelForCausalLM consume.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_DIR = Path(__file__).resolve().parent
REPO_ID = "Junfeng5/Liquid_V1_7B"

snapshot_download(
    repo_id=REPO_ID,
    local_dir=MODEL_DIR,
    # trainer_state.json & README.md of the upstream repo also land
    # here; they are harmless metadata for inference.
)

print(f"Liquid_V1_7B downloaded to {MODEL_DIR}")
