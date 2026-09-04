import asyncio
# store inference result
_inference_queue = None

loop = None
# data_channel = None

def get_inference_queue():
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = asyncio.Queue(maxsize=30)
    return _inference_queue

liquid_bridge = None
def get_liquid_bridge(enc_dec):
    global liquid_bridge
    if liquid_bridge is None:
        from aiortc.codecs.liquid_tokenizer import liquidBridge
        liquid_bridge = liquidBridge(enc_dec)
    return liquid_bridge


