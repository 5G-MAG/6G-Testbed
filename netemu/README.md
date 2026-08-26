# netemu

Linux network emulation, packet capture, and packet-level metric measurement for Python.

`netemu` covers the whole measurement loop for a shaped network:

| Stage | Module | What it does |
|:------|:-------|:-------------|
| **Emulate** | `netemu.emulator` | Drives `tc`/`netem`/HTB to impose delay, jitter, loss, rate limits, corruption, reordering and duplication, on egress and (via IFB) on ingress |
| **Capture** | `netemu.capture` | Runs `tcpdump` on the shaped interface, writing a pcap plus a JSON metadata sidecar |
| **Analyze** | `netemu.pcap` | Parses that pcap back into network-layer metrics: throughput, RTT, retransmissions, TLS setup, per-direction volumes, burstiness, burst segmentation |
| **Measure** | `netemu.metrics` | Aggregates one or many parsed captures into distributions: connection setup, flow lifetime, per-direction volume, burst structure, windowed throughput, retransmissions |

The four stages are independent. You can shape without capturing, analyze
captures produced by any other tool, aggregate captures the emulator never
touched, or use the emulator alone.

Everything the last two stages report is derived from packet headers alone:
sizes, timestamps, directions, TCP flags and sequence numbers. No module here
knows about application payloads, test scenarios, traffic labels, or any
external metric taxonomy.

## Features

### Emulation

- **Latency & jitter**: delay with configurable distributions (normal, pareto, paretonormal)
- **Packet loss**: random, correlated, or Gilbert-Elliot (`gemodel`)
- **Bandwidth limiting**: HTB-based rate limiting with burst/cburst and ceil
- **Corruption**: single-bit error injection
- **Reordering** and **duplication**, each with correlation
- **Bidirectional shaping**: egress directly, ingress redirected through an IFB device
- **Asymmetric links**: different profiles for uplink and downlink
- **Profile-based configuration**: YAML profile files
- **Context manager**: automatic cleanup on exit

### Capture

- `tcpdump` lifecycle management with startup validation (bad interface, bad BPF, missing permissions are detected instead of silently producing an empty file)
- Packet-buffered writes (`-U`) so a capture is readable while it runs
- JSON metadata sidecar per capture: interface, BPF filter, start/end wall-clock time, size, plus arbitrary caller-supplied fields
- Output validation on stop: exit code and pcap header are checked before the file is reported as good
- `capture_to()` context manager that stops the capture even if the body raises

### PCAP analysis

- libpcap and pcapng input
- IPv4 and IPv6; TCP and UDP
- Bidirectional TCP flow reassembly with connection lifecycle (SYN, SYN-ACK, ACK, first data, FIN, RST)
- Handshake RTT, TLS handshake duration, time-to-first-data, data-transfer duration
- Retransmission detection in per-direction sequence space
- Per-packet records for sub-second analysis
- Multi-window throughput (1 ms to 10 s) and burstiness
- Per-direction burst segmentation and inter-burst idle gaps

### Packet metrics

- One call aggregates a single capture or a whole campaign into distributions
- Connection metrics: handshake RTT, TLS setup, full connection setup, time to first data, flow duration, exchanges per flow, connection reuse, destination fan-out
- Direction metrics: uplink and downlink volumes and their ratio, per-direction packet size and inter-packet gap
- Burst metrics: count, size, duration, peak rate and inter-burst idle, per gap threshold and direction
- Throughput metrics: mean and peak, plus peak and burstiness per averaging window
- Reliability metrics: retransmission counts and rates, per capture and per flow
- Every distribution carries `n`, mean, min, max, p50, p95, p99, stdev and coefficient of variation
- `to_dict()` on any report is JSON-serializable

## Requirements

- **Python 3.10+**
- **Linux** with `iproute2` (`tc`) for emulation
- **`tcpdump`** for capture
- **Sudo access** for `tc`, `ip`, `modprobe`, and `tcpdump`
- **Kernel modules**: `sch_netem`, `sch_htb`, `ifb` (for bidirectional shaping)
- **`dpkt`** for pcap analysis (optional extra)

## Installation

netemu lives inside the [5G-MAG/6G-Testbed](https://github.com/5G-MAG/6G-Testbed)
repository but is a self-contained package: it has its own `pyproject.toml`,
license, and tests, and imports nothing from the rest of the testbed.

From a local checkout:

```bash
# Emulation + capture only
pip install -e /path/to/netemu

# Including pcap analysis (pulls in dpkt)
pip install -e "/path/to/netemu[pcap]"
```

To install netemu without cloning the testbed, pip can install straight from
the repository subdirectory:

```bash
pip install "netemu[pcap] @ git+https://github.com/5G-MAG/6G-Testbed.git#subdirectory=netemu"
```

To work on the netemu source without checking out the rest of the testbed,
use a sparse checkout, which materializes only this directory:

```bash
git clone --filter=blob:none --sparse https://github.com/5G-MAG/6G-Testbed.git
cd 6G-Testbed
git sparse-checkout set netemu
pip install -e "netemu[pcap]"
```

Importing `netemu` without `dpkt` succeeds. `netemu.HAS_DPKT` is then `False`,
the pcap names are `None`, and constructing a `PcapAnalyzer` raises
`DpktNotAvailableError` (which also subclasses `ImportError`).

## Quick Start

### Emulate

```python
from netemu import NetworkEmulator

emulator = NetworkEmulator(interface="eth0")
emulator.apply_settings(delay_ms=100, loss_pct=1.0, rate_mbit=10)

# ... run your tests ...

emulator.clear()
```

Context manager (automatic cleanup):

```python
with NetworkEmulator(interface="eth0") as emu:
    emu.apply_settings(delay_ms=50, jitter_ms=20)
    # ... run your tests ...
# Rules automatically cleared on exit
```

Profiles with asymmetric uplink/downlink:

```python
emulator = NetworkEmulator(interface="eth0", profiles_path="profiles.yaml")

# Egress/uplink: good_wifi (30 ms, 50 Mbps)
# Ingress/downlink: poor_cellular (120 ms, 5 Mbps)
emulator.apply_profile("good_wifi", ingress_profile="poor_cellular")
```

### Capture and analyze

```python
from netemu import NetworkEmulator, capture_to, analyze_pcap

with NetworkEmulator(interface="eth0") as emu:
    emu.apply_profile("cell_edge")
    with capture_to("run.pcap", interface="eth0", filter_expr="port 443"):
        run_workload()

m = analyze_pcap("run.pcap", target_ports=[443])

print(f"{m.total_packets} packets in {m.tcp_flows} TCP flows")
print(f"handshake RTT: mean {m.rtt_mean_ms:.1f} ms, p95 {m.rtt_p95_ms:.1f} ms")
print(f"retransmissions: {m.retransmission_rate:.2%}")
print(f"UL/DL bytes: {m.ul_bytes_total} / {m.dl_bytes_total}")
print(f"burstiness @100ms: {m.burstiness_by_window['100ms']:.2f}")
```

### Measure a campaign

```python
from netemu import analyze_multiple_pcaps, compute_packet_metrics

captures = analyze_multiple_pcaps("captures/", target_ports=[443, 8443])
report = compute_packet_metrics(captures)

print(report.connection.connection_setup_ms.p95)   # tail setup cost, ms
print(report.direction.ul_dl_byte_ratio)           # uplink / downlink bytes
print(report.reliability.flow_retransmission_ratio)

import json
json.dump(report.to_dict(), open("metrics.json", "w"), indent=2)
```

A single capture works the same way: `compute_packet_metrics(analyze_pcap(path))`.

---

# PCAP Processing

`netemu.pcap` turns a capture file into numbers. This section documents what
is parsed, how each metric is computed, and where the definitions have edges
worth knowing about.

## Pipeline

```
   pcap / pcapng file
          │
          ▼
   ┌──────────────────────────────────────────────────┐
   │ 1. Frame decode      Ethernet -> IPv4 / IPv6     │
   │                      -> TCP / UDP                │
   ├──────────────────────────────────────────────────┤
   │ 2. Port filter       optional target_ports,      │
   │                      with per-file exemptions    │
   ├──────────────────────────────────────────────────┤
   │ 3. Per-packet record timestamp, size, direction, │
   │                      TCP flags, seq/ack, window  │
   ├──────────────────────────────────────────────────┤
   │ 4. Flow state        bidirectional TCP flows:    │
   │                      handshake, TLS, FIN/RST,    │
   │                      retransmissions             │
   ├──────────────────────────────────────────────────┤
   │ 5. Aggregation       RTT, throughput, retrans    │
   ├──────────────────────────────────────────────────┤
   │ 6. Post-processing   per-direction totals,       │
   │                      multi-window throughput,    │
   │                      burst segmentation          │
   └──────────────────────────────────────────────────┘
          │
          ▼
      PcapMetrics
```

Steps 1 to 5 happen in a single pass over the file. Step 6
(`_compute_per_direction_and_multi_window`) runs afterwards over the
per-packet records that step 3 accumulated, so it never re-reads the file.

## Entry points

```python
from netemu.pcap import (
    PcapAnalyzer,            # the analyzer object
    analyze_pcap,            # one file  -> PcapMetrics
    analyze_multiple_pcaps,  # directory -> list[PcapMetrics]
    merge_pcap_metrics,      # list[PcapMetrics] -> aggregate dict
)
```

```python
PcapAnalyzer(
    target_ports: list[int] | None = None,
    *,
    unfiltered_name_patterns: Sequence[str] = (),
)

PcapAnalyzer.analyze(pcap_path: str, bucket_sec: float = 1.0) -> PcapMetrics
```

| Parameter | Meaning |
|:----------|:--------|
| `target_ports` | Keep only packets whose source or destination port is in this set. `None` analyzes every port. Also used to disambiguate traffic direction (see below) |
| `unfiltered_name_patterns` | File-name substrings that exempt a capture from `target_ports`. Needed for loopback captures, where local servers bind ports that are not known when the filter is configured |
| `bucket_sec` | Bucket width for the main `throughput_timeseries`. Does not affect the fixed multi-window series |

`analyze_multiple_pcaps()` logs and skips files that fail to parse, so one
truncated capture does not abort a batch.

## Direction attribution

Every packet is labelled `"ul"` (uplink, client to server) or `"dl"`
(downlink, server to client). Two rules, in order:

1. **Port-based.** If `target_ports` is set and exactly one of the source or
   destination port is in it, the port in the set is the server. A packet
   *towards* it is `ul`; a packet *from* it is `dl`.
2. **Fallback heuristic.** Otherwise the lower port number is assumed to be
   the server: `sport < dport` gives `dl`, else `ul`.

The fallback is a heuristic and is wrong for connections between two
ephemeral ports. Pass `target_ports` whenever the server port is known: it is
what makes direction attribution on loopback captures reliable.

## Sizes: which byte count is which

Three different byte counts appear in `PcapMetrics`, and they are not
interchangeable:

| Field | Counts |
|:------|:-------|
| `total_bytes`, `tcp_bytes`, `udp_bytes` | Full captured frame, including the Ethernet header |
| `PacketRecord.size`, `ul_bytes_total`, `dl_bytes_total` | IP payload, i.e. the TCP segment or UDP datagram including its transport header |
| `PacketRecord.payload_len` | Transport payload only, excluding the TCP/UDP header |

`avg_throughput_mbps` is derived from `total_bytes` (frame level). The
per-direction and windowed series are derived from `PacketRecord.size`
(IP-payload level). Comparing the two directly will show a constant offset
of roughly the Ethernet header per packet.

## TCP flow reassembly

Packets are grouped into bidirectional flows. The first packet seen for a
4-tuple establishes the forward direction and the flow key
`src_ip:src_port-dst_ip:dst_port`; packets matching the reversed tuple join
the same flow as the reverse direction.

Each `TCPFlow` records:

| Field | Meaning |
|:------|:--------|
| `packets_sent` / `packets_recv` | Packet counts in the forward / reverse direction |
| `bytes_sent` / `bytes_recv` | IP-payload bytes per direction |
| `start_time` / `end_time` | First and last packet timestamp |
| `syn_time` | First SYN without ACK |
| `syn_ack_time` | First SYN+ACK |
| `ack_time` | First bare ACK after a SYN-ACK (handshake completion) |
| `first_data_time` | First packet carrying transport payload |
| `fin_time` / `rst_time` | First FIN / first RST |
| `retransmissions` | Count of payload byte ranges seen more than once |
| `tls_client_hello_time` | First TLS handshake record (type `0x16`) |
| `tls_first_app_data_time` | First TLS application-data record (type `0x17`) |

Derived properties:

| Property | Definition |
|:---------|:-----------|
| `duration` | `end_time - start_time`, or 0 |
| `handshake_rtt` | `ack_time - syn_time`, the full three-way handshake |
| `syn_ack_rtt` | `syn_ack_time - syn_time`, one round trip to the server |
| `time_to_first_data` | `first_data_time - syn_time`, handshake plus TLS setup |
| `tls_handshake_duration` | `tls_first_app_data_time - tls_client_hello_time` |
| `data_transfer_duration` | `first_data_time` to `fin_time` / `rst_time` / `end_time` |
| `throughput_bps` | `(bytes_sent + bytes_recv) * 8 / duration` |
| `ul_throughput_bps` / `dl_throughput_bps` | Same, per direction |
| `retransmission_rate` | `retransmissions / (packets_sent + packets_recv)` |

All timing properties return `None` when the required markers were not
captured, which is the normal case for a flow that was already open when the
capture started.

### Retransmission detection

A packet with payload contributes the sequence interval
`[seq, seq + payload_len)`. The interval is merged into a list of intervals
already seen **in that direction**; if it overlaps an existing interval the
retransmission counter increments.

Two consequences worth knowing:

- **Pure ACKs are ignored.** They carry no payload, so a repeated ACK is not
  counted as a retransmission.
- **The two directions have separate sequence spaces.** TCP sequence numbers
  are independent per direction; tracking a single set would make an equal
  client and server sequence number look like a retransmission.

This is a byte-range overlap test, not a full TCP state machine. It counts
retransmitted *data*, and will also count the payload of a TCP fast-retransmit
or a spurious retransmission. It does not detect reordering separately.

### TLS handshake detection

TLS is identified by record-type sniffing on the first bytes of each TCP
payload, without decryption. A payload is treated as starting a TLS record
when byte 1 is `0x03` and byte 2 is in `0x01..0x04` (TLS 1.0 to 1.3). Then:

- record type `0x16` (Handshake), first occurrence, is taken as the ClientHello
- record type `0x17` (ApplicationData), first occurrence, marks the handshake complete

The interval between them bounds the full handshake for both TLS 1.2 and
TLS 1.3. Because this is a heuristic on unencrypted record headers, it can
miss a handshake that is already in progress when the capture starts, and it
does not distinguish a resumed session from a full handshake.

## Aggregate metrics

### RTT

`rtt_samples` collects `handshake_rtt` from every flow that completed a
three-way handshake, converted to milliseconds. From those:

```
rtt_mean_ms = mean(rtt_samples)
rtt_min_ms  = min(rtt_samples)
rtt_max_ms  = max(rtt_samples)
rtt_p95_ms  = sorted(rtt_samples)[min(int(len * 0.95), len - 1)]
```

The p95 uses nearest-rank on the sorted list, not interpolation. RTT here is
**connection-setup RTT only**. It is a clean signal because the handshake is
unencrypted and unambiguous, but it samples the network once per connection.
On a connection-reusing protocol like HTTP/2 there will be far fewer RTT
samples than requests.

### Retransmissions

```
total_retransmissions = sum(flow.retransmissions for flow in flows)
retransmission_rate   = total_retransmissions / tcp_packets
```

The denominator is all TCP packets, including pure ACKs, so the rate is
lower than a retransmitted-data-packets-over-data-packets figure would be.

### Throughput

```
avg_throughput_mbps = total_bytes * 8 / capture_duration / 1e6
```

`throughput_timeseries` is a list of `(rel_time_sec, ul_kbps, dl_kbps)` at
`bucket_sec` resolution (default 1 s), and `peak_throughput_mbps` is the
largest `ul + dl` across those buckets.

### Per-direction totals

Computed from the per-packet records:

| Field | Definition |
|:------|:-----------|
| `ul_packets` / `dl_packets` | Packet count per direction |
| `ul_bytes_total` / `dl_bytes_total` | IP-payload bytes per direction |
| `ul_mean_pkt_size` / `dl_mean_pkt_size` | Bytes divided by packets, `None` if that direction is empty |

The UL/DL byte ratio, the headline number for asymmetry, is
`ul_bytes_total / dl_bytes_total`.

### Multi-window throughput and burstiness

The same packets are re-bucketed at five fixed window widths: **1 ms, 10 ms,
100 ms, 1 s, 10 s**. For each window label:

```python
throughput_by_window[label]   # [(rel_time_sec, ul_bps, dl_bps), ...]
peak_mbps_by_window[label]    # max(ul + dl) across buckets, in Mbps
burstiness_by_window[label]   # max(ul + dl) / mean(ul + dl)
```

Burstiness is a **peak-to-mean ratio**, so it is 1.0 for perfectly smooth
traffic and grows without bound as traffic concentrates. It is strongly
window-dependent: the same flow looks far burstier at 1 ms than at 10 s,
which is exactly why all five windows are reported rather than one.

Note that buckets with no packets are absent from the series rather than
present as zeros. The mean is therefore taken over *active* buckets, which
makes burstiness a measure of variation within the active period rather than
of duty cycle. Use the burst segmentation below to characterize idleness.

### Burst segmentation

Packets are split by direction, sorted by time, and cut wherever the
inter-arrival time exceeds a gap threshold. Two thresholds are computed:
**10 ms** and **100 ms**.

```python
bursts_by_gap["100ms"]["ul"]           # list of bursts
interburst_idle_by_gap["100ms"]["ul"]  # list of idle-gap durations in seconds
```

Each burst is a dict:

| Key | Meaning |
|:----|:--------|
| `start`, `end` | Absolute timestamps of the first and last packet |
| `duration_sec` | `end - start`; **zero for a single-packet burst** |
| `total_bytes` | Sum of IP-payload sizes |
| `packet_count` | Packets in the burst |
| `peak_rate_bps` | `total_bytes * 8 / duration_sec` |

When `duration_sec` is zero the rate would be undefined, so `peak_rate_bps`
falls back to `total_bytes * 8 / gap_sec`, which reads as "at least this
fast". Treat single-packet bursts as rate lower bounds.

`interburst_idle_by_gap` holds the inter-arrival times that triggered each
cut, so its length is one less than the burst count for that direction. Its
coefficient of variation is a useful measure of how regular the traffic's
on/off pattern is.

## Merging captures

`merge_pcap_metrics(list[PcapMetrics]) -> dict` aggregates a batch:

```python
{
  "total_captures", "total_packets", "total_bytes", "total_duration_sec",
  "total_tcp_flows", "total_retransmissions",
  "avg_throughput_mbps",                       # total_bytes over summed duration
  "rtt_mean_ms", "rtt_min_ms", "rtt_max_ms", "rtt_p95_ms",
  "retransmission_rate",
  "throughput_timeseries",                     # concatenated, sorted by time
}
```

`total_duration_sec` is the **sum** of the individual capture durations, not
the wall-clock span they cover. For captures taken in parallel the merged
`avg_throughput_mbps` is therefore an underestimate. RTT samples are pooled
across all captures before the percentile is taken.

## Data model reference

```python
@dataclass
class PacketRecord:
    timestamp: float     # absolute capture time, seconds
    size: int            # IP payload length (transport header + payload)
    direction: str       # "ul" or "dl"
    tcp_flags: int = 0
    seq: int = 0
    ack: int = 0
    window: int = 0
    payload_len: int = 0 # transport payload only
    flow_key: str = ""
```

`PcapMetrics.packets` holds one of these per accepted packet, which is what
makes sub-second and packet-by-packet analysis possible downstream. It is
also the largest thing in memory: a multi-gigabyte capture will hold millions
of these. For long captures, analyze per-file and merge, rather than
concatenating pcaps first.

`PcapMetrics.to_dict()` returns the scalar summary fields only (no packet
lists, no time series), which is what you want for JSON reports.

---

# Packet Metrics

`netemu.metrics` turns parsed captures into the distributions a measurement
campaign reports. `netemu.pcap` answers "what happened in this capture";
`netemu.metrics` answers "what does this set of captures look like".

## Entry point

```python
from netemu import compute_packet_metrics

report = compute_packet_metrics(captures)   # one PcapMetrics, or any iterable
```

`captures` accepts a single `PcapMetrics`, a list, or a generator. The return is
a `PacketMetricsReport`. Metric families with no supporting data come back empty
rather than absent, so the shape of a report never depends on what a particular
capture happened to contain.

## Distributions

Every distribution is a `Distribution`:

| Field | Meaning |
|:------|:--------|
| `n` | Sample size after dropping NaN and infinities |
| `mean`, `stdev`, `cv` | Mean, population standard deviation, and their ratio |
| `minimum`, `maximum` | Extremes |
| `p50`, `p95`, `p99` | Nearest-rank percentiles |

Percentiles use nearest-rank on the sorted sample, so every reported value is an
observed value rather than an interpolation between two. An empty sample gives
`n = 0` and `None` everywhere else, which is distinguishable from a real zero.

`summarize(values)` is exported for callers that want the same summary over their
own samples.

## Sampling unit

Two units appear, and mixing them up is the easiest way to misread a report:

- **Pooled over flows.** `tcp_handshake_rtt_ms`, `tls_handshake_ms`,
  `connection_setup_ms`, `flow_duration_s`, `exchanges_per_flow` treat each flow
  as one observation, pooled across every capture.
- **Per capture.** Anything named `*_per_capture` treats each capture as one
  observation. This is the right unit when captures differ in length, since a
  long capture would otherwise dominate a pooled sample.

## Metric families

### Connection

| Field | Meaning |
|:------|:--------|
| `flows` | Total TCP flows across the set |
| `tcp_handshake_rtt_ms` | SYN to SYN-ACK |
| `tls_handshake_ms` | TLS ClientHello to first application data |
| `connection_setup_ms` | SYN to first application data: handshake, any local scheduling delay, and TLS when present |
| `time_to_first_data_ms` | SYN to the first payload-bearing segment |
| `flow_duration_s` | First to last packet of a flow |
| `flows_per_capture` | Flow count per capture |
| `exchanges_per_flow` | Application request-response turns per flow |
| `reuse_ratio` | Fraction of flows carrying at least two exchanges |
| `distinct_destinations_per_capture` | Server-side fan-out |

`exchanges_per_flow` counts uplink-to-downlink alternations between
payload-bearing packets, which is as close to a request-response turn as packet
headers allow. Two details make it trustworthy:

- Empty segments are skipped, so pure ACKs never count as a turn.
- Counting starts at each flow's **first application data**. A TLS handshake is
  itself a payload-bearing exchange in both directions, and including it
  inflated the count by roughly 20 packets per flow on a sampled capture, which
  in turn pushed `reuse_ratio` from a plausible 0.4% to an impossible 95%.

`reuse_ratio` therefore answers whether connections are genuinely reused or
opened for a single turn and discarded.

### Direction

| Field | Meaning |
|:------|:--------|
| `ul_bytes`, `dl_bytes`, `ul_packets`, `dl_packets` | Totals across the set |
| `ul_dl_byte_ratio`, `ul_dl_packet_ratio` | Uplink over downlink; `None` when the denominator is zero |
| `ul_packet_size`, `dl_packet_size` | Per-direction packet size distribution |
| `ul_inter_packet_gap_ms`, `dl_inter_packet_gap_ms` | Gap between consecutive packets in one direction |
| `ul_dl_byte_ratio_per_capture` | The ratio computed per capture, then summarized |

Byte counts follow the same convention as `netemu.pcap`: IP payload, so the TCP
header and payload but not the IP or Ethernet headers. See "Sizes: which byte
count is which" above.

### Burst

`bursts.by_gap[gap_label][direction]` holds `count`, `size_bytes`, `duration_s`,
`peak_rate_bps` and `idle_gap_s`, each a serialized `Distribution`. Gap labels
are whatever thresholds `PcapAnalyzer` was configured with, `10ms` and `100ms`
by default. `count` is per capture; the rest pool over individual bursts.

### Throughput

`mean_mbps` and `peak_mbps` are per capture. `peak_mbps_by_window` and
`burstiness_by_window` are keyed by averaging window (`1ms` through `10s`), each
value a distribution over captures. Burstiness is peak over mean within a
window, so it is at least 1 by construction, and it grows as the averaging
window shrinks toward the packet timescale.

### Reliability

| Field | Meaning |
|:------|:--------|
| `total_retransmissions` | Sum across the set |
| `retransmission_rate_per_capture` | Percentage per capture |
| `retransmission_rate_per_flow` | Percentage per flow |
| `flows_with_retransmissions` | Count of affected flows |
| `flow_retransmission_ratio` | Affected flows over total flows |

The per-capture and per-flow rates answer different questions. A capture rate
near zero with a high flow ratio means retransmissions are spread thinly across
many flows rather than concentrated in a few.

## Serialization

`report.to_dict()` returns nested plain dicts and is JSON-serializable, with
distributions flattened to `{"n", "mean", "min", "max", "p50", "p95", "p99",
"stdev", "cv"}`. Note the dict keys are `min` and `max` where the dataclass
fields are `minimum` and `maximum`, to keep the JSON conventional.

## Scope

This module derives everything from packet headers. It has no notion of
application payload, test scenario, traffic class, or reporting taxonomy. Joining
these metrics to labels, scenarios, or application-layer measurements belongs in
the caller, which keeps the package reusable for any shaped-network measurement.

# Packet Capture

```python
from netemu import CaptureController, capture_to
```

## CaptureController

```python
CaptureController(interface: str = "eth0", capture_dir: str = "captures")
```

| Method | Description |
|:-------|:------------|
| `start(filename=None, filter_expr=None, metadata=None)` | Start tcpdump; returns the pcap path or `None` on failure |
| `stop()` | Stop tcpdump, validate output, write the sidecar; returns the path or `None` |
| `is_running()` | Whether a capture is currently active |
| `get_capture_stats(pcap_file)` | Summary via `capinfos` if installed, else file size |

`start()` invokes:

```
sudo tcpdump -i <interface> -w <file> -U [<filter_expr>]
```

`-U` is packet-buffered output, so the file is readable while the capture
runs. `filter_expr` is a BPF expression (`"port 443"`), shell-split before
being appended.

`start()` waits 250 ms and re-checks the process, because `Popen` succeeding
only means `sudo`/`tcpdump` was executed. A bad interface, an invalid BPF
expression, or missing sudo rights all cause tcpdump to exit immediately;
without the re-check they would surface much later as a mysteriously empty
pcap. On that path `start()` logs tcpdump's stderr and returns `None`.

`stop()` refuses to report success unless the exit code is 0 or -15 (SIGTERM,
the normal path) and the file exists with at least a 24-byte pcap header.

## Metadata sidecar

Every successful `stop()` writes `<pcap>.metadata.json` alongside the capture,
mode `0600`:

```json
{
  "schema_version": 1,
  "pcap_file": "capture_20260807_101500.pcap",
  "interface": "eth0",
  "filter": "port 443",
  "t_start": 1786000000.123,
  "t_end":   1786000042.456,
  "size_bytes": 1048576
}
```

Anything passed as `metadata=` to `start()` is merged in, which is how a
capture gets attributed to whatever produced it (scenario name, profile,
run index). `t_start`/`t_end` are wall-clock bounds, so a capture can be
intersected with an externally recorded activity window: this is what lets
persistent connections be segmented per experiment rather than being
attributed wholesale to whichever run happened to open them.

## capture_to()

```python
with capture_to("run.pcap", interface="eth0", filter_expr="port 443") as cap:
    run_workload()
# capture stopped, sidecar written, even if run_workload() raised
```

## Scope

`netemu.capture` is L3/L4 only. For decrypted L7 (HTTP request/response
bodies) use a TLS-terminating proxy such as mitmproxy; that requires trusting
an interception certificate in the client and is outside netemu's scope.

---

# Emulation API Reference

### NetworkEmulator

```python
NetworkEmulator(
    interface: str = "eth0",          # or "auto" to detect the default route
    profiles_path: Optional[str] = None,
    bidirectional: bool = True,
    ifb_device: str = "ifb0"
)
```

**Parameters:**
- `interface`: Network interface to apply rules to. `"auto"` detects the interface carrying the default route
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

An interface passed explicitly to the constructor always wins over
`default_interface` in a profiles file. The YAML value applies only when the
constructor was given `interface="auto"`.

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
| `DpktNotAvailableError` | pcap analysis attempted without `dpkt`; also an `ImportError` |

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

Capturing on the shaped interface sees traffic *after* egress shaping and
*before* ingress shaping is undone, so the pcap reflects the emulated
conditions. Note that shaping and capture interact: a rate limit delays
packets before they reach the wire, so the timestamps in a capture on the
shaping host already include the emulated delay.

## Sudoers Setup

netemu requires root privileges for `tc`, `ip`, `modprobe`, and (for capture)
`tcpdump`. To run without password prompts, create a sudoers drop-in file:

```bash
sudo visudo -f /etc/sudoers.d/netemu
```

Add the following line (adjust the binary paths to match your system):

```
username ALL=(ALL) NOPASSWD: /usr/sbin/tc, /usr/sbin/ip, /usr/sbin/modprobe, /usr/bin/tcpdump
```

Or for a group:

```
%netemu ALL=(ALL) NOPASSWD: /usr/sbin/tc, /usr/sbin/ip, /usr/sbin/modprobe, /usr/bin/tcpdump
```

Binary paths vary by distribution. Find the correct paths with:

```bash
which tc ip modprobe tcpdump
```

Common locations:
- Debian/Ubuntu/WSL2: `/usr/sbin/tc`, `/usr/sbin/ip`, `/usr/sbin/modprobe`, `/usr/bin/tcpdump`
- Older distros: `/sbin/tc`, `/sbin/ip`, `/sbin/modprobe`

Verify passwordless access:

```bash
sudo -n tc qdisc show dev eth0
sudo -n ip link show
sudo -n modprobe -n ifb
sudo -n tcpdump --version
```

All four should complete without prompting for a password.

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

### Capture starts but the pcap stays empty

Usually a BPF filter that matches nothing, or the wrong interface. Check that
`start()` did not return `None` (it validates tcpdump's startup), then verify
the filter by hand:

```bash
sudo tcpdump -i eth0 -c 5 port 443
```

### Analysis reports zero packets

Either the `target_ports` filter excluded everything, or the capture is on
loopback where local servers use ports the filter does not know about. Pass
`unfiltered_name_patterns` for such captures, or drop `target_ports`.

### `DpktNotAvailableError` on PcapAnalyzer

Install the extra: `pip install "netemu[pcap]"`.

## Shell Scripts

Standalone shell scripts are provided in `scripts/`:

```bash
# Apply basic profile
./scripts/apply_profile.sh eth0 100 1.0 10  # 100ms delay, 1% loss, 10Mbps

# Clear all rules
./scripts/clear_profile.sh eth0
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite is hermetic: tc and tcpdump are never executed, and the pcap tests
synthesize capture files with dpkt rather than shipping binary fixtures.

## License

Apache-2.0. See [LICENSE](LICENSE).
