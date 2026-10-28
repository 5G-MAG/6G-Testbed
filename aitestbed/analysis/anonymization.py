"""
Anonymization helpers for provider, model, and scenario identifiers.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


DEFAULT_MAP_PATH = Path(__file__).resolve().parent.parent / "configs" / "anonymization_map.json"


class Anonymizer:
    """Map provider/model/scenario identifiers to stable aliases."""

    def __init__(self, map_path: Optional[str] = None, strict: Optional[bool] = None):
        map_path = map_path or os.environ.get("ANONYMIZATION_MAP_PATH") or str(DEFAULT_MAP_PATH)
        self.map_path = Path(map_path)
        self._mapping = self._load_mapping(self.map_path)
        self._providers = self._mapping.get("providers", {})
        self._models = self._mapping.get("models", {})
        self._scenarios = self._mapping.get("scenarios", {})
        self._defaults = self._mapping.get("defaults", {})

        if strict is None:
            strict = os.environ.get("ANONYMIZATION_STRICT", "0") == "1"
        self._strict = bool(strict)

        self._provider_aliases = set(self._providers.values())
        self._model_aliases = set(self._models.values())
        self._scenario_aliases = set(self._scenarios.values())

    def provider_alias(self, provider: Optional[str]) -> Optional[str]:
        """Return anonymized provider alias."""
        return self._alias(provider, self._providers, self._provider_aliases, "provider")

    def model_alias(self, model: Optional[str]) -> Optional[str]:
        """Return anonymized model alias."""
        return self._alias(model, self._models, self._model_aliases, "model")

    def scenario_alias(self, scenario: Optional[str]) -> Optional[str]:
        """Return anonymized scenario alias."""
        return self._alias(scenario, self._scenarios, self._scenario_aliases, "scenario")

    def _alias(self, value: Optional[str], mapping: dict, aliases: set, category: str) -> Optional[str]:
        if value is None or value == "":
            return value
        if value in aliases:
            return value
        if value in mapping:
            return mapping[value]
        if self._strict:
            raise KeyError(f"Missing anonymization mapping for {category}: {value}")
        return self._defaults.get(category, f"unknown_{category}")

    @staticmethod
    def _load_mapping(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Anonymization map not found: {path}")
        with path.open("r") as f:
            return json.load(f)


@lru_cache(maxsize=1)
def get_anonymizer(map_path: Optional[str] = None, strict: Optional[bool] = None) -> Anonymizer:
    """Return a cached anonymizer instance."""
    return Anonymizer(map_path=map_path, strict=strict)
