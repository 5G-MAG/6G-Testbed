# training

Markov traffic-pattern generators learned from the captures the sibling
`aitestbed/` testbed produces: category-conditioned categorical models that
synthesize flows per traffic pattern, so a study can be driven without
re-running a live measurement campaign.

This component reads `aitestbed` **output files** only. It imports no testbed
code, so it can be pointed at any capture set with the same layout.

## Layout

```
training/
├── dataset.py            # pcap + SQLite -> features / sequences / labels
├── quantization.py       # shared size/IAT bin codec + empirical dequantizer
├── train_markov.py       # order-0 and order-1 categorical generators
├── generator_metrics.py  # held-out KS metrics and comparison plots
├── requirements.txt
├── tests/                # dataset, codec, and generator unit tests
├── data/                 # generated: features_k*.npz, sequences.npz, labels.json
├── models/               # generated: markov/
├── synthetic/            # generated: per-category synthetic flows
└── results/              # generated: metrics, plots
```

`data/`, `models/`, `synthetic/`, and `results/` are all **generated**. They
are not shipped; the commands below rebuild them from the captures.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# 1. build the dataset from the testbed captures + measurement log
python -m dataset     --captures-dir ../aitestbed/results/captures     --db-path ../aitestbed/logs/traffic_logs.db     --output-dir data --max-packets 100

# 2. fit and evaluate the Markov generators
python -m train_markov --data-dir data
```

`train_markov.py` writes the fitted tables to `models/markov/model.npz`,
per-category synthetic flows to `synthetic/`, and held-out metrics to
`results/generator_markov/`.

---

# The dataset

`dataset.py` turns raw captures plus the measurement log into three artifacts:
per-flow feature vectors at several values of *k*, per-flow packet sequences for
the generator, and a label file carrying the split assignment and provenance.

## Sources and attribution

Three inputs are consumed:

- packet captures, including the loopback captures for scenarios that have a
  local control plane.
- the measurement log, which carries the scenario, network profile, run index,
  and the exact start and end time of every session.
- the per-capture metadata sidecar written next to each pcap, which carries
  scenario, profile, and run provenance directly.

Both `--captures-dir` and `--db-path` accept several values or globs, so a run
can span the live output and every archived cycle. Archived databases are
cumulative snapshots of one another, so sessions are deduplicated by
`session_id` and pcaps by basename plus size.

Attribution proceeds per flow:

- **Transport surface.** A flow whose two endpoints are both loopback addresses
  is the agent's local control plane. Everything else is WAN egress. WAN flows
  are kept only on recognized service ports (443, 80, 8080, 8443) to exclude
  unrelated host traffic. Loopback captures are kept on every port, because
  local servers bind anywhere: MCP on 8765, A2A agents on 9001 to 9003, the
  OpenClaw gateway on 18789, vLLM on 8000, and ephemeral ports elsewhere.
- **Direction.** The client and server side of each TCP flow are resolved from
  the handshake rather than from port numbers. Port-based direction is
  meaningless on loopback, where a local server binds above the client's
  ephemeral range.
- **Session boundaries.** Each flow is intersected with the exact session window
  from the log. A persistent connection spanning two consecutive experiments is
  segmented at the boundary, so packets from one run cannot carry the label of
  another.
- **Protocols.** IPv4, IPv6, TCP, and UDP are parsed. WebRTC is retained only
  when media UDP is present. Its TCP and TLS signaling is not relabeled as
  media.

The testbed's default BPF captures WebRTC's negotiated UDP transport through the
ICE, STUN, and TURN service ports 3478, 3479, 5349, 5350, and 19302. Captures
made with an older TCP-only filter cannot contribute `realtime_webrtc` samples
and need to be recaptured.

## Traffic categories

Nine labels, defined in `dataset.py::SCENARIO_CATEGORY` and `CATEGORY_IDS`:

| label | what it looks like on the wire |
|-------|-------------------------------|
| `request_response` | single request, single non-streamed response |
| `streaming` | single request, chunked or streamed response (SSE, long-poll) |
| `agent_loop` | iterative LLM and tool-call rounds, growing context, mixed UL/DL |
| `realtime_ws` | persistent WebSocket, bidirectional, low latency |
| `realtime_webrtc` | WebRTC media over UDP, sustained bandwidth |
| `bulk_transfer` | large asymmetric payload: images, video, binaries |
| `parallel_burst` | fan-out of many concurrent HTTP requests plus synthesis |
| `agent_signaling` | agent-to-agent protocol messages (A2A JSON-RPC `message/send` and SSE `message/stream`): small near-deterministic payloads, about one round trip per interaction, no LLM in the path |
| `agent_control` | local control channel between an agent runtime and a local tool or gateway server (MCP JSON-RPC, OpenClaw gateway): loopback, unencrypted JSON, small requests with large local responses |

`agent_signaling` covers all three A2A variants. The streaming variant differs
by a few SSE chunks inside the same sub-second exchange, which is not separable
at packet level.

`agent_control` is assigned by transport surface rather than by scenario name.
Any `agent_loop` scenario contributes `agent_control` flows on loopback and
`agent_loop` flows on its WAN egress, through `LOOPBACK_CATEGORY_OVERRIDE`. This
keeps the OpenClaw gateway channel on `lo:18789` separate from the cloud LLM
egress of the same session. The two surfaces have opposite asymmetry and very
different packet dynamics, and one label across both would teach a downstream
model a contradiction.

## Splitting

Flows are partitioned 70/15/15 into training, validation, and test at the level
of the **capture group**, not the individual flow. A capture group covers one
capture and its loopback pair, so no capture, session, or sample can appear in
more than one partition.

`_stratified_group_split()` searches group assignments under a fixed seed,
scoring each candidate against the requested per-class proportions and applying
a heavy penalty when a feasible class is missing from a split. The chosen
assignment is asserted disjoint across the three partitions.

This is the step that determines whether the reported scores mean anything. A
split at flow level places flows from the same capture on both sides of the
boundary and inflates every metric.

## Feature vector

`compute_aggregate_features()` reduces the first *k* packets of a flow to 26
features, emitted for *k* in {5, 10, 15, 20, 30, 100}:

- **Size**: mean and standard deviation per direction, maximum, entropy over
  size bins, payload ratio.
- **Timing**: mean, standard deviation, and minimum inter-arrival time, its
  coefficient of variation, burst count, time to first downlink packet,
  downlink packet count within one second.
- **Direction and interaction**: uplink and downlink byte ratio, uplink packet
  fraction, direction-switch count and rate, mean run length, longest run in one
  direction, round-trip count, mean downlink burst size.
- **Transport**: total bytes, PSH-flag fraction, mean TCP window, window growth
  rate.

The interaction features exist to separate `agent_loop`, which alternates
direction frequently, from `streaming`, which produces long downlink runs.

## Current snapshot

| quantity | value |
|----------|-------|
| Attributed flows | 56,297 |
| Independent capture groups | 927 |
| Source pcaps | 1,960 |
| Labeled sessions | 8,022 |
| TCP / UDP flows | 56,133 / 164 |
| Loopback flows | 19,729 |
| Train / validation / test | 40,369 / 8,460 / 7,468 |

Class support is heavily imbalanced:

| class | flows | | class | flows |
|-------|------:|-|-------|------:|
| `streaming` | 16,259 | | `bulk_transfer` | 4,868 |
| `agent_loop` | 9,399 | | `request_response` | 3,672 |
| `agent_control` | 9,235 | | `agent_signaling` | 471 |
| `parallel_burst` | 6,770 | | `realtime_webrtc` | 9 |
| `realtime_ws` | 5,614 | | | |

`realtime_webrtc` holds nine flows across six training, two validation, and one
test capture group. That is enough to keep the class in every grouped split and
in the exported label map. It is not enough for a per-class estimate. The same
caution applies to `agent_signaling`, which holds roughly 67 test flows.

---

---

# Generator design

## Flow representation

A flow is a sequence of packets, each carrying three channels, plus one
flow-level quantity:

- packet size, quantized to 65 bins.
- inter-arrival time, quantized to 65 logarithmically spaced bins.
- direction, as a binary uplink or downlink indicator.
- sequence length, up to 100 packets.

Sizes and inter-arrival times are heavy tailed and span several orders of
magnitude. Modeling them as categorical distributions over bins preserves the
multimodal structure that separates one category from another. A regression
formulation collapses toward the mean and loses it.

## Markov baselines

`train_markov.py` fits two category-conditioned categorical generators:

- **Order 0** samples the per-category marginal of each channel independently
  per packet. It reproduces the per-packet marginals by construction. It carries
  no memory, so it represents no relationship between one packet and the next.
- **Order 1** samples channel-specific transition matrices, with a
  category-marginal Dirichlet backoff for states too sparse to estimate. It
  captures short-range structure inside each channel. The three channels remain
  independent chains, with no shared state and no flow-level state.

Both draw sequence length from the empirical per-category training distribution
and decode through the shared codec. They fit the clean-profile training
partition and are scored on the held-out partition.

These are not strawmen. They are the honest floor: a learned generator that
cannot beat an empirical sampler has not earned its complexity.

## Dequantization codec

The generators emit bin indices, which are decoded back to bytes and
seconds. Emitting the bin center replaces a range of values by a single point,
which puts a floor of roughly 0.24 to 0.46 on the per-category size KS statistic
regardless of generator quality. A generator that
emits bin centers scores a mean size KS of about 0.39 on this dataset, and
that figure is dominated by decoding loss rather than model error.

`EmpiricalDequantizer` instead stores, per category and bin, the distribution of
raw values observed inside that bin as a quantile grid, and samples it by
inverse CDF. It is fitted once on the clean training partition and shared
unchanged by every generator that uses the codec, so a comparison measures
generation rather than decoding.

## Evaluation protocol

Each generator is scored per category with the two-sample Kolmogorov-Smirnov
statistic against 931 held-out clean-profile flows across the seven categories
with sufficient test support. Each generator produces `max(512, n_real)`
synthetic flows per category, so the statistic is not dominated by
synthetic-side sampling noise. Reported values are unweighted category means,
and lower is better.

Six quantities are compared, in two groups that behave differently:

- **Per-packet marginals**: size and inter-arrival time. These describe packets
  in isolation.
- **Per-flow joint statistics**: length, uplink fraction, total bytes, duration.
  These depend on how the packets of a flow relate to one another.

Per-flow size autocorrelation at lag 1 is reported separately as a direct probe
of whether packet-to-packet structure survives generation.

## Testing

```bash
python -m pytest tests
```

The tests cover flow attribution, session segmentation, capture-group leakage,
feature extraction, the dequantization codec, and seeded Markov generation. They require neither captures nor a GPU.
