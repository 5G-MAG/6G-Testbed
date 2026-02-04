# netemu

A Python wrapper for Linux tc/netem network emulation.

`netemu` provides a simple Python interface to emulate various network conditions including latency, jitter, packet loss, bandwidth limiting, corruption, reordering, and duplication. It's useful for testing applications under different network conditions.

## Features

- **Latency & Jitter**: Add delay with configurable distributions (normal, pareto, uniform)
- **Packet Loss**: Random or correlated loss with advanced models (Gilbert-Elliot)
- **Bandwidth Limiting**: HTB-based rate limiting with burst support
- **Corruption**: Single-bit error injection
- **Reordering**: Packet reordering emulation
- **Duplication**: Packet duplication
- **Bidirectional Shaping**: Shape both ingress and egress traffic using IFB devices
- **Profile-based Configuration**: Load network profiles from YAML files
- **Context Manager**: Automatic cleanup on exit

## Requirements

- **Python 3.10+**
- **Linux** with `iproute2` (tc command)
- **Sudo access** for tc commands
- **Kernel modules**: `sch_netem`, `sch_htb`, `ifb` (for bidirectional shaping)

## Installation

```bash
# From local directory
pip install -e /path/to/netemu

```

## Quick Start

```python
from netemu import NetworkEmulator

# Apply custom settings
emulator = NetworkEmulator(interface="eth0")
emulator.apply_settings(delay_ms=100, loss_pct=1.0, rate_mbit=10)

# ... run your tests ...

emulator.clear()
```

Using profiles with asymmetric uplink/downlink:

```python
from netemu import NetworkEmulator

emulator = NetworkEmulator(
    interface="eth0",
    profiles_path="profiles.yaml"
)

# Apply different profiles for uplink (egress) and downlink (ingress)
# Simulates asymmetric link: fast upload, slow download
emulator.apply_profile(
    "good_wifi",           # Egress/uplink: 30ms, 50 Mbps
    ingress_profile="poor_cellular"  # Ingress/downlink: 120ms, 5 Mbps
)

# ... run your tests ...

emulator.clear()
```

Context manager (automatic cleanup):

```python
from netemu import NetworkEmulator

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_settings(delay_ms=50, jitter_ms=20)
    # ... run your tests ...
# Rules automatically cleared on exit
```

## API Reference

### NetworkEmulator

```python
NetworkEmulator(
    interface: str = "eth0",
    profiles_path: Optional[str] = None,
    bidirectional: bool = True,
    ifb_device: str = "ifb0"
)
```

**Parameters:**
- `interface`: Network interface to apply rules to
- `profiles_path`: Path to YAML file with profile definitions
- `bidirectional`: If True, shape both egress and ingress traffic
- `ifb_device`: IFB device name for ingress shaping

**Methods:**

| Method | Description |
|--------|-------------|
| `apply_profile(name, ingress_profile=None)` | Apply a named profile |
| `apply_settings(**kwargs)` | Apply custom network settings |
| `clear()` | Remove all tc rules |
| `load_profiles(path)` | Load profiles from YAML file |
| `list_profiles()` | Get list of available profile names |
| `get_profile(name)` | Get a profile by name |
| `get_status()` | Get current tc/netem status |
| `check_sudo()` | Check if passwordless sudo is available |

### NetworkProfile

```python
@dataclass
class NetworkProfile:
    name: str
    description: str = ""
    delay_ms: int = 0
    jitter_ms: int = 0
    delay_distribution: Optional[str] = None  # normal, pareto, paretonormal
    delay_correlation_pct: Optional[float] = None
    loss_pct: float = 0.0
    loss_correlation_pct: Optional[float] = None
    loss_model: Optional[str] = None  # gemodel, state
    rate_mbit: Optional[int] = None
    rate_ceil_mbit: Optional[int] = None
    rate_burst_kbit: Optional[int] = None
    rate_cburst_kbit: Optional[int] = None
    corruption_pct: float = 0.0
    corruption_correlation_pct: Optional[float] = None
    reorder_pct: float = 0.0
    reorder_correlation_pct: Optional[float] = None
    duplicate_pct: float = 0.0
    duplicate_correlation_pct: Optional[float] = None
    limit_packets: Optional[int] = None
```

### Exceptions

| Exception | Description |
|-----------|-------------|
| `NetEmuError` | Base exception for all netemu errors |
| `SudoNotAvailableError` | Sudo access required but not available |
| `ProfileNotFoundError` | Requested profile not found |
| `CommandFailedError` | tc/ip command failed |
| `ProfileLoadError` | Profile file cannot be loaded |

## Configuration

### Profile YAML Format

```yaml
profiles:
  profile_name:
    description: "Human readable description"
    delay_ms: 100
    jitter_ms: 20
    delay_distribution: normal
    loss_pct: 1.0
    rate_mbit: 10

default_interface: "eth0"
bidirectional: true
```

### Example Profiles

See `examples/profiles.yaml` for a complete set of example profiles including:
- `ideal` - No impairments
- `good_wifi` - Typical WiFi conditions
- `poor_cellular` - Poor cellular/cell edge
- `satellite` - High latency satellite link
- `congested` - Heavy congestion

## Bidirectional Shaping

By default, netemu shapes both egress (outbound) and ingress (inbound) traffic using IFB (Intermediate Functional Block) devices.

```python
# Same profile for both directions
emulator.apply_profile("slow_network")

# Different profiles for each direction
emulator.apply_profile("fast_upload", ingress_profile="slow_download")

# Egress only (disable ingress shaping)
emulator.apply_profile("slow_network", ingress_profile="none")

# Or disable at initialization
emulator = NetworkEmulator(interface="eth0", bidirectional=False)
```

## Sudoers Setup

To run without password prompts, add to `/etc/sudoers` (use `visudo`):

```
# Allow user to run tc and ip commands without password
username ALL=(ALL) NOPASSWD: /sbin/tc, /sbin/ip, /sbin/modprobe ifb*
```

Or for a group:

```
%netemu ALL=(ALL) NOPASSWD: /sbin/tc, /sbin/ip, /sbin/modprobe ifb*
```

## Troubleshooting

### "RTNETLINK answers: No such file or directory"

This error occurs when trying to delete rules that don't exist. It's safe to ignore and is handled internally.

### "Operation not permitted"

Ensure you have sudo access configured. Run `sudo tc qdisc show` to verify permissions.

### IFB device not available

Load the IFB kernel module:

```bash
sudo modprobe ifb numifbs=1
sudo ip link set dev ifb0 up
```

### Netem module not loaded

```bash
sudo modprobe sch_netem
```

### Changes not taking effect

Verify rules are applied:

```bash
tc qdisc show dev eth0
tc class show dev eth0
```

## Shell Scripts

Standalone shell scripts are provided in `scripts/`:

```bash
# Apply basic profile
./scripts/apply_profile.sh eth0 100 1.0 10  # 100ms delay, 1% loss, 10Mbps

# Clear all rules
./scripts/clear_profile.sh eth0
```


