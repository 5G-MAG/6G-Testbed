"""Pytest configuration and fixtures for netemu tests."""

import pytest


@pytest.fixture
def sample_profile_data():
    """Sample profile data for testing."""
    return {
        "description": "Test profile",
        "delay_ms": 100,
        "jitter_ms": 20,
        "loss_pct": 1.0,
        "rate_mbit": 10,
    }


@pytest.fixture
def sample_profiles_yaml(tmp_path):
    """Create a temporary profiles YAML file."""
    content = """
profiles:
  test_profile:
    description: "Test profile"
    delay_ms: 100
    jitter_ms: 20
    loss_pct: 1.0
    rate_mbit: 10

  ideal:
    description: "No impairments"
    delay_ms: 0
    loss_pct: 0

default_interface: "eth0"
"""
    profiles_file = tmp_path / "profiles.yaml"
    profiles_file.write_text(content)
    return str(profiles_file)
