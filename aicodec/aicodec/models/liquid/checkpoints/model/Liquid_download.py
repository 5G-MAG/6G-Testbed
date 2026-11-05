from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "/home/pcan1032/6G-Testbed/aiortc-main-clean/src/aiortc/liquid/checkpoints/model"

tokenizer = AutoTokenizer.from_pretrained("Junfeng5/Liquid_V1_7B", cache_dir=MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained("Junfeng5/Liquid_V1_7B", cache_dir=MODEL_DIR)
