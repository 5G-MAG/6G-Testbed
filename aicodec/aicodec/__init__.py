"""aicodec — AI codec plugin for a stock aiortc install.

Registers the token-based AI codecs (``video/tokenId``, ``video/tokenIdTransparent``,
``video/tokenIdLLM`` + rtx) into aiortc at runtime, with loss-tolerant framing,
and provides a pluggable VLM model layer behind them.

Usage:
    import aicodec
    aicodec.register()            # once, before creating RTCPeerConnection
    aicodec.set_model("liquid")   # optional: select VLM model (default "liquid")

    from aicodec.codec import TokenIdEncoder, AsyncTokenIdDecoder
    import aicodec.shared as shared   # loop / inference queue / model_name

New VLM models: subclass :class:`aicodec.models.VLMBridge` and
:func:`~aicodec.models.register_model`; see ``aicodec/models/base.py``.
"""

from . import shared
from .register import register, is_registered
from .codec import (
    TokenIdEncoder,
    TokenIdDecoder,
    AsyncTokenIdDecoder,
    set_bridge_model,
    get_bridge_model,
)
from .models import VLMBridge, register_model, available_models, get_bridge

set_model = shared.set_model
get_model = shared.get_model

__version__ = "1.0.0"

__all__ = [
    "register",
    "is_registered",
    "shared",
    "set_model",
    "get_model",
    "set_bridge_model",
    "get_bridge_model",
    "TokenIdEncoder",
    "TokenIdDecoder",
    "AsyncTokenIdDecoder",
    # model layer
    "VLMBridge",
    "register_model",
    "available_models",
    "get_bridge",
]
