"""Tests for NetworkProfile dataclass."""

from netemu import NetworkProfile


def test_profile_defaults():
    """Test NetworkProfile default values."""
    profile = NetworkProfile(name="test")

    assert profile.name == "test"
    assert profile.description == ""
    assert profile.delay_ms == 0
    assert profile.jitter_ms == 0
    assert profile.loss_pct == 0.0
    assert profile.rate_mbit is None
    assert profile.corruption_pct == 0.0
    assert profile.reorder_pct == 0.0
    assert profile.duplicate_pct == 0.0


def test_profile_from_dict(sample_profile_data):
    """Test NetworkProfile.from_dict factory method."""
    profile = NetworkProfile.from_dict("my_profile", sample_profile_data)

    assert profile.name == "my_profile"
    assert profile.description == "Test profile"
    assert profile.delay_ms == 100
    assert profile.jitter_ms == 20
    assert profile.loss_pct == 1.0
    assert profile.rate_mbit == 10


def test_profile_from_dict_missing_fields():
    """Test from_dict with minimal data."""
    profile = NetworkProfile.from_dict("minimal", {})

    assert profile.name == "minimal"
    assert profile.description == ""
    assert profile.delay_ms == 0
    assert profile.loss_pct == 0.0


def test_profile_from_dict_all_fields():
    """Test from_dict with all fields specified."""
    data = {
        "description": "Full profile",
        "delay_ms": 50,
        "jitter_ms": 10,
        "delay_distribution": "normal",
        "delay_correlation_pct": 25.0,
        "loss_pct": 2.0,
        "loss_correlation_pct": 30.0,
        "loss_model": "gemodel",
        "rate_mbit": 100,
        "rate_ceil_mbit": 150,
        "rate_burst_kbit": 1500,
        "rate_cburst_kbit": 1500,
        "corruption_pct": 0.1,
        "corruption_correlation_pct": 10.0,
        "reorder_pct": 5.0,
        "reorder_correlation_pct": 50.0,
        "duplicate_pct": 0.5,
        "duplicate_correlation_pct": 20.0,
        "limit_packets": 1000,
    }

    profile = NetworkProfile.from_dict("full", data)

    assert profile.delay_distribution == "normal"
    assert profile.delay_correlation_pct == 25.0
    assert profile.loss_model == "gemodel"
    assert profile.rate_ceil_mbit == 150
    assert profile.corruption_pct == 0.1
    assert profile.limit_packets == 1000
