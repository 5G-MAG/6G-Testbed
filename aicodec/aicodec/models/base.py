"""Abstract base class for pluggable VLM model bridges.

A bridge provides the model-specific backend used by the token codecs:
image tokenization, detokenization, multimodal LLM generation and token
counting.  Third-party models (liquid, chameleon, ...) each live in a
sub-package under ``aicodec.models`` and register themselves via
``register_model(name, bridge_cls)``.

To add a new model:
  1. create ``aicodec/models/<name>/`` with a ``bridge.py`` subclassing
     :class:`VLMBridge`
  2. call ``register_model("<name>", <BridgeCls>)`` in the sub-package's
     ``__init__.py``
  3. put checkpoints under ``aicodec/models/<name>/checkpoints/``
  4. select it from the testbed config (``model: <name>`` in scenarios.yaml)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional


class VLMBridge(ABC):
    """Model-side backend for the token codecs.

    ``enc_dec`` selects what to construct (matches the legacy liquid bridge):
      - ``"enc"`` / ``"dec"``: image tokenizer only (client / detokenizer)
      - ``"llm"``: image tokenizer + multimodal LLM (full inference)
    """

    #: name used as the result-JSON key in benchmark output
    model_name: str = ""

    def __init__(self, enc_dec: str) -> None:
        self.enc_dec = enc_dec
        self.device = self._resolve_device()
        self.setup()

    def _resolve_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def checkpoints_path(self) -> Path:
        return Path(__file__).parent / self.model_name / "checkpoints"

    @abstractmethod
    def setup(self) -> None:
        """Load checkpoints / weights according to self.enc_dec."""

    @abstractmethod
    def tokenize(self, image) -> list[int]:
        """PIL image -> token id sequence."""

    @abstractmethod
    def detokenize(self, tokens) -> object:
        """Token id sequence -> PIL image."""

    @abstractmethod
    def generate(self, image_tokens, user_prompt: str) -> Iterator[str]:
        """Multimodal generation: stream text chunks for the given tokens."""

    @abstractmethod
    def estimate_token_number(self, text: str) -> int:
        """Number of text tokens (for metrics)."""
