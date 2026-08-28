"""Tests for netemu.capture (tcpdump-driven packet capture).

tcpdump is never actually executed; subprocess.Popen is patched so the
controller's lifecycle, metadata sidecar, and failure handling can be
verified without root or a live interface.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from netemu.capture import CaptureController, capture_to

PCAP_HEADER = b"\xd4\xc3\xb2\xa1" + b"\x00" * 20  # 24 bytes: a minimal pcap header


def _fake_proc(poll_return=None, returncode=0):
    proc = MagicMock()
    proc.poll.return_value = poll_return
    proc.returncode = returncode
    proc.stderr.read.return_value = b""
    return proc


@pytest.fixture
def capture_dir(tmp_path):
    d = tmp_path / "captures"
    d.mkdir()
    return d


class TestInit:
    def test_creates_capture_dir(self, tmp_path):
        target = tmp_path / "nested" / "captures"
        CaptureController(interface="eth0", capture_dir=str(target))

        assert target.is_dir()

    def test_defaults(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        assert cap.interface == "eth0"
        assert cap.is_running() is False


class TestStart:
    def test_builds_tcpdump_command(self, capture_dir):
        cap = CaptureController(interface="wlan0", capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()) as popen:
            with patch("netemu.capture.time.sleep"):
                path = cap.start(filename="run.pcap", filter_expr="port 443")

        cmd = popen.call_args[0][0]
        assert cmd[:2] == ["sudo", "tcpdump"]
        assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "wlan0"
        assert "-U" in cmd                       # packet-buffered
        assert cmd[-2:] == ["port", "443"]       # BPF filter, shell-split
        assert path == capture_dir / "run.pcap"

    def test_autogenerates_filename(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()):
            with patch("netemu.capture.time.sleep"):
                path = cap.start()

        assert path.name.startswith("capture_")
        assert path.suffix == ".pcap"

    def test_detects_immediate_tcpdump_exit(self, capture_dir):
        """A bad interface or invalid BPF makes tcpdump die right away."""
        cap = CaptureController(capture_dir=str(capture_dir))
        proc = _fake_proc(poll_return=1)
        proc.stderr.read.return_value = b"tcpdump: no such device"

        with patch("netemu.capture.subprocess.Popen", return_value=proc):
            with patch("netemu.capture.time.sleep"):
                result = cap.start(filename="run.pcap")

        assert result is None
        assert cap.is_running() is False

    def test_second_start_is_a_noop(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()) as popen:
            with patch("netemu.capture.time.sleep"):
                first = cap.start(filename="a.pcap")
                second = cap.start(filename="b.pcap")

        assert popen.call_count == 1
        assert second == first

    def test_popen_failure_returns_none(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.Popen", side_effect=OSError("boom")):
            assert cap.start(filename="run.pcap") is None


class TestStop:
    def _start(self, cap, filename="run.pcap", metadata=None):
        proc = _fake_proc()
        with patch("netemu.capture.subprocess.Popen", return_value=proc):
            with patch("netemu.capture.time.sleep"):
                cap.start(filename=filename, filter_expr="port 443", metadata=metadata)
        return proc

    def test_stop_without_start(self, capture_dir):
        assert CaptureController(capture_dir=str(capture_dir)).stop() is None

    def test_writes_metadata_sidecar(self, capture_dir):
        cap = CaptureController(interface="eth0", capture_dir=str(capture_dir))
        self._start(cap, metadata={"scenario": "chat_basic", "profile": "5g_urban"})
        (capture_dir / "run.pcap").write_bytes(PCAP_HEADER)

        result = cap.stop()

        assert result == capture_dir / "run.pcap"
        sidecar = capture_dir / "run.pcap.metadata.json"
        meta = json.loads(sidecar.read_text())

        assert meta["schema_version"] == 1
        assert meta["pcap_file"] == "run.pcap"
        assert meta["interface"] == "eth0"
        assert meta["filter"] == "port 443"
        assert meta["scenario"] == "chat_basic"
        assert meta["profile"] == "5g_urban"
        assert meta["t_end"] >= meta["t_start"]
        assert meta["size_bytes"] == len(PCAP_HEADER)

    def test_sidecar_is_owner_only(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))
        self._start(cap)
        (capture_dir / "run.pcap").write_bytes(PCAP_HEADER)
        cap.stop()

        mode = (capture_dir / "run.pcap.metadata.json").stat().st_mode & 0o777
        assert mode == 0o600

    def test_truncated_pcap_rejected(self, capture_dir):
        """A file shorter than a pcap header means tcpdump wrote nothing."""
        cap = CaptureController(capture_dir=str(capture_dir))
        self._start(cap)
        (capture_dir / "run.pcap").write_bytes(b"short")

        assert cap.stop() is None

    def test_missing_file_rejected(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))
        self._start(cap)

        assert cap.stop() is None

    def test_sigterm_exit_is_success(self, capture_dir):
        """terminate() makes tcpdump exit with -15; that is the normal path."""
        cap = CaptureController(capture_dir=str(capture_dir))
        proc = self._start(cap)
        proc.returncode = -15
        (capture_dir / "run.pcap").write_bytes(PCAP_HEADER)

        assert cap.stop() == capture_dir / "run.pcap"

    def test_nonzero_exit_rejected(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))
        proc = self._start(cap)
        proc.returncode = 2
        (capture_dir / "run.pcap").write_bytes(PCAP_HEADER)

        assert cap.stop() is None

    def test_kills_unresponsive_tcpdump(self, capture_dir):
        import subprocess as sp

        cap = CaptureController(capture_dir=str(capture_dir))
        proc = self._start(cap)
        proc.wait.side_effect = [sp.TimeoutExpired(cmd="tcpdump", timeout=5), 0]
        (capture_dir / "run.pcap").write_bytes(PCAP_HEADER)

        cap.stop()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestIsRunning:
    def test_true_while_process_alive(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()):
            with patch("netemu.capture.time.sleep"):
                cap.start(filename="run.pcap")

        assert cap.is_running() is True


class TestCaptureStats:
    def test_falls_back_to_file_size(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))
        pcap = capture_dir / "run.pcap"
        pcap.write_bytes(PCAP_HEADER)

        with patch("netemu.capture.subprocess.run", side_effect=FileNotFoundError):
            stats = cap.get_capture_stats(pcap)

        assert stats["size_bytes"] == len(PCAP_HEADER)

    def test_parses_capinfos_output(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))
        pcap = capture_dir / "run.pcap"
        pcap.write_bytes(PCAP_HEADER)

        result = MagicMock(returncode=0,
                           stdout="Number of packets: 42\nCapture duration: 1.5 seconds\n")
        with patch("netemu.capture.subprocess.run", return_value=result):
            stats = cap.get_capture_stats(pcap)

        assert stats["Number of packets"] == "42"
        assert stats["Capture duration"] == "1.5 seconds"

    def test_missing_file_returns_empty(self, capture_dir):
        cap = CaptureController(capture_dir=str(capture_dir))

        with patch("netemu.capture.subprocess.run", side_effect=FileNotFoundError):
            assert cap.get_capture_stats(capture_dir / "gone.pcap") == {}


class TestCaptureToContextManager:
    def test_stops_on_normal_exit(self, capture_dir):
        target = capture_dir / "ctx.pcap"

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()):
            with patch("netemu.capture.time.sleep"):
                with capture_to(str(target), interface="eth0", filter_expr="port 443") as cap:
                    assert cap.is_running() is True
                    target.write_bytes(PCAP_HEADER)

        assert cap.is_running() is False

    def test_stops_even_when_body_raises(self, capture_dir):
        target = capture_dir / "ctx.pcap"

        with patch("netemu.capture.subprocess.Popen", return_value=_fake_proc()):
            with patch("netemu.capture.time.sleep"):
                with pytest.raises(RuntimeError):
                    with capture_to(str(target), interface="eth0") as cap:
                        target.write_bytes(PCAP_HEADER)
                        raise RuntimeError("workload failed")

        assert cap.is_running() is False
