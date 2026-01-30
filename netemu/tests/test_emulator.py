"""Tests for NetworkEmulator class."""

import pytest

from netemu import NetworkEmulator, NetworkProfile
from netemu.exceptions import ProfileLoadError, ProfileNotFoundError


class TestNetworkEmulatorInit:
    """Tests for NetworkEmulator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        emu = NetworkEmulator()

        assert emu.interface == "eth0"
        assert emu.bidirectional is True
        assert emu.ifb_device == "ifb0"
        assert emu.profiles == {}
        assert emu.current_profile is None

    def test_custom_interface(self):
        """Test initialization with custom interface."""
        emu = NetworkEmulator(interface="enp0s3")

        assert emu.interface == "enp0s3"

    def test_disable_bidirectional(self):
        """Test disabling bidirectional shaping."""
        emu = NetworkEmulator(bidirectional=False)

        assert emu.bidirectional is False


class TestProfileLoading:
    """Tests for profile loading functionality."""

    def test_load_profiles(self, sample_profiles_yaml):
        """Test loading profiles from YAML file."""
        emu = NetworkEmulator()
        emu.load_profiles(sample_profiles_yaml)

        assert len(emu.profiles) == 2
        assert "test_profile" in emu.profiles
        assert "ideal" in emu.profiles

        profile = emu.profiles["test_profile"]
        assert profile.delay_ms == 100
        assert profile.loss_pct == 1.0

    def test_load_profiles_in_constructor(self, sample_profiles_yaml):
        """Test loading profiles during construction."""
        emu = NetworkEmulator(profiles_path=sample_profiles_yaml)

        assert len(emu.profiles) == 2

    def test_load_profiles_sets_interface(self, sample_profiles_yaml):
        """Test that default_interface from YAML is applied."""
        emu = NetworkEmulator(interface="original")
        emu.load_profiles(sample_profiles_yaml)

        assert emu.interface == "eth0"

    def test_load_profiles_file_not_found(self):
        """Test error when profile file doesn't exist."""
        emu = NetworkEmulator()

        with pytest.raises(ProfileLoadError) as exc_info:
            emu.load_profiles("/nonexistent/path.yaml")

        assert "file not found" in str(exc_info.value)

    def test_load_profiles_invalid_yaml(self, tmp_path):
        """Test error on invalid YAML."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{invalid yaml")

        emu = NetworkEmulator()

        with pytest.raises(ProfileLoadError) as exc_info:
            emu.load_profiles(str(bad_yaml))

        assert "invalid YAML" in str(exc_info.value)

    def test_load_profiles_empty_file(self, tmp_path):
        """Test error on empty file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        emu = NetworkEmulator()

        with pytest.raises(ProfileLoadError) as exc_info:
            emu.load_profiles(str(empty_file))

        assert "empty file" in str(exc_info.value)


class TestProfileAccess:
    """Tests for profile access methods."""

    def test_list_profiles(self, sample_profiles_yaml):
        """Test listing profile names."""
        emu = NetworkEmulator(profiles_path=sample_profiles_yaml)

        profiles = emu.list_profiles()

        assert "test_profile" in profiles
        assert "ideal" in profiles

    def test_get_profile(self, sample_profiles_yaml):
        """Test getting a profile by name."""
        emu = NetworkEmulator(profiles_path=sample_profiles_yaml)

        profile = emu.get_profile("test_profile")

        assert profile is not None
        assert profile.name == "test_profile"
        assert profile.delay_ms == 100

    def test_get_profile_not_found(self, sample_profiles_yaml):
        """Test getting non-existent profile returns None."""
        emu = NetworkEmulator(profiles_path=sample_profiles_yaml)

        profile = emu.get_profile("nonexistent")

        assert profile is None


class TestNetemParams:
    """Tests for netem parameter building."""

    def test_build_netem_params_delay(self):
        """Test building delay parameters."""
        emu = NetworkEmulator()

        params = emu._build_netem_params(
            delay_ms=100,
            jitter_ms=0,
            delay_distribution=None,
            delay_correlation_pct=None,
            loss_pct=0,
            loss_correlation_pct=None,
            loss_model=None,
            corruption_pct=0,
            corruption_correlation_pct=None,
            reorder_pct=0,
            reorder_correlation_pct=None,
            duplicate_pct=0,
            duplicate_correlation_pct=None,
            limit_packets=None,
        )

        assert "delay 100ms" in params

    def test_build_netem_params_jitter(self):
        """Test building delay with jitter parameters."""
        emu = NetworkEmulator()

        params = emu._build_netem_params(
            delay_ms=100,
            jitter_ms=20,
            delay_distribution="normal",
            delay_correlation_pct=25.0,
            loss_pct=0,
            loss_correlation_pct=None,
            loss_model=None,
            corruption_pct=0,
            corruption_correlation_pct=None,
            reorder_pct=0,
            reorder_correlation_pct=None,
            duplicate_pct=0,
            duplicate_correlation_pct=None,
            limit_packets=None,
        )

        param_str = " ".join(params)
        assert "delay 100ms" in param_str
        assert "20ms" in param_str
        assert "25.0%" in param_str
        assert "distribution normal" in param_str

    def test_build_netem_params_loss(self):
        """Test building loss parameters."""
        emu = NetworkEmulator()

        params = emu._build_netem_params(
            delay_ms=0,
            jitter_ms=0,
            delay_distribution=None,
            delay_correlation_pct=None,
            loss_pct=5.0,
            loss_correlation_pct=25.0,
            loss_model=None,
            corruption_pct=0,
            corruption_correlation_pct=None,
            reorder_pct=0,
            reorder_correlation_pct=None,
            duplicate_pct=0,
            duplicate_correlation_pct=None,
            limit_packets=None,
        )

        param_str = " ".join(params)
        assert "loss 5.0%" in param_str
        assert "25.0%" in param_str

    def test_build_netem_params_empty(self):
        """Test building params with no impairments."""
        emu = NetworkEmulator()

        params = emu._build_netem_params(
            delay_ms=0,
            jitter_ms=0,
            delay_distribution=None,
            delay_correlation_pct=None,
            loss_pct=0,
            loss_correlation_pct=None,
            loss_model=None,
            corruption_pct=0,
            corruption_correlation_pct=None,
            reorder_pct=0,
            reorder_correlation_pct=None,
            duplicate_pct=0,
            duplicate_correlation_pct=None,
            limit_packets=None,
        )

        assert params == []


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_enter(self):
        """Test entering context manager returns self."""
        emu = NetworkEmulator()

        with emu as ctx:
            assert ctx is emu

    def test_context_manager_clears_on_exit(self, mocker):
        """Test that clear() is called on context exit."""
        emu = NetworkEmulator()
        mock_clear = mocker.patch.object(emu, 'clear')

        with emu:
            pass

        mock_clear.assert_called_once()
