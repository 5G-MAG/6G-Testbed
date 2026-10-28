"""Model registry: pluggable VLM bridges keyed by model name.

Built-in models self-register on import.  External models can register at
runtime:

    from aicodec.models import register_model
    register_model("mymodel", MyBridge)
"""

from typing import Callable, Optional, Type

from .base import VLMBridge

_REGISTRY: dict[str, Type[VLMBridge]] = {}

_BRIDGE_CACHE: dict[tuple[str, str], VLMBridge] = {}


def register_model(name: str, bridge_cls: Type[VLMBridge]) -> None:
    """Register a bridge class under ``name`` (later wins, idempotent)."""
    bridge_cls.model_name = name
    _REGISTRY[name] = bridge_cls


def available_models() -> list[str]:
    _load_builtin_models()
    return sorted(_REGISTRY.keys())


def get_bridge_cls(name: str) -> Type[VLMBridge]:
    _load_builtin_models()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown model '{name}'; available: {sorted(_REGISTRY.keys())}"
        ) from None


def get_bridge(name: str, enc_dec: str, reload: bool = False) -> VLMBridge:
    """Instantiate (and cache) the bridge for ``name``.

    ``enc_dec``: "enc" | "dec" | "llm" (see VLMBridge).
    ``reload``: drop the cached instance first (fresh weights).
    """
    _load_builtin_models()
    key = (name, enc_dec)
    if reload:
        _BRIDGE_CACHE.pop(key, None)
    if key not in _BRIDGE_CACHE:
        cls = get_bridge_cls(name)
        _BRIDGE_CACHE[key] = cls(enc_dec)
    return _BRIDGE_CACHE[key]


_BUILTIN_LOADED = False


def _load_builtin_models() -> None:
    global _BUILTIN_LOADED
    if _BUILTIN_LOADED:
        return
    # import side effect: registers "liquid"
    try:
        from . import liquid  # noqa: F401
    except ImportError:
        # heavy deps (torch/transformers) not installed in this environment
        pass
    _BUILTIN_LOADED = True


__all__ = [
    "VLMBridge",
    "register_model",
    "available_models",
    "get_bridge_cls",
    "get_bridge",
]
