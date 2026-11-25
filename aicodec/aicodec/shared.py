"""Shared state for aicodec (inference queue, event loop, selected model).

Model bridges are resolved through :mod:`aicodec.models` by name; this module
only keeps process-wide state shared between the codecs and consumers.
"""

import asyncio

# store inference result
_inference_queue = None

loop = None
# data_channel = None

# selected model name — single source of truth for the codec layer;
# set via aicodec.set_model() / set_bridge_model() /
# AsyncTokenIdDecoder.reset_decoder_state(model_name=...)
model_name = "liquid"

def get_inference_queue():
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = asyncio.Queue(maxsize=30)
    return _inference_queue

def set_model(name: str) -> None:
    """Select the model used when resolving bridges by name."""
    global model_name
    model_name = name

def get_model() -> str:
    return model_name
