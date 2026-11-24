"""Liquid model package: registers the liquid VLM bridge."""

from .. import register_model
from .bridge import LiquidBridge

register_model("liquid", LiquidBridge)

__all__ = ["LiquidBridge"]
