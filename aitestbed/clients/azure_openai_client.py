"""
Azure OpenAI client adapter for the 6G AI Traffic Testbed.

Subclasses OpenAIClient since the openai SDK's AzureOpenAI class returns
identical response objects — all parent methods (chat, streaming, image
generation, tool calling) work unchanged.
"""

import os
from typing import Optional

from openai import AzureOpenAI

from .openai_client import OpenAIClient


class AzureOpenAIClient(OpenAIClient):
    """
    Azure OpenAI API client adapter.

    Uses the same openai SDK but connects to an Azure-hosted deployment.
    The ``model`` parameter in scenario YAML corresponds to the Azure
    deployment name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        # Skip OpenAIClient.__init__ — we build the client ourselves.
        self.client = AzureOpenAI(
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_version=api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        self._last_request_bytes = 0
        self._last_response_bytes = 0

    @property
    def provider(self) -> str:
        return "azure_openai"
