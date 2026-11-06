# METRICS.md

Detailed documentation of all metrics calculated by the 6G AI Traffic Characterization Testbed.

## Table of Contents

1. [Overview](#overview)
2. [Data Capture Layer](#data-capture-layer)
3. [Metric Categories](#metric-categories)
4. [Calculation Methods](#calculation-methods)
5. [3GPP Output Format](#3gpp-output-format)
6. [Configuration Parameters](#configuration-parameters)
7. [RAN2 Methodology Metrics (S4-260859)](#ran2-methodology-metrics-s4-260859)

---

## Overview

The testbed calculates metrics at multiple levels:

1. **Per-Request Level** - Raw measurements captured during each API call (`LogRecord`)
2. **Aggregated Level** - Statistical summaries per scenario/profile combination (`ScenarioMetrics`)
3. **3GPP Report Level** - Formatted output for inclusion in 3GPP SA4 6G Media Study report

### Data Flow

```
API Call → LogRecord → SQLite Storage → MetricsCalculator → ScenarioMetrics → 3GPP Report
```

---

## Data Capture Layer

### LogRecord (`analysis/logger.py`)

The foundational data structure capturing raw metrics per API request.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | When the record was created |
| `scenario_id` | string | Identifier for the scenario |
| `session_id` | string | Unique session identifier |
| `turn_index` | int | Turn number within multi-turn conversation |
| `run_index` | int | Experiment run number |
| `request_bytes` | int | Size of request payload in bytes |
| `response_bytes` | int | Size of response payload in bytes |
| `tokens_in` | int | Input/prompt token count |
| `tokens_out` | int | Output/completion token count |
| `t_request_start` | float | Absolute timestamp when request started |
| `t_first_token` | float | Timestamp of first token/chunk arrival |
| `t_last_token` | float | Timestamp of last token/chunk arrival |
| `latency_sec` | float | Total request-to-response time in seconds |
| `http_status` | int | HTTP status code (200, 429, 500, etc.) |
| `error_type` | string | Type of error encountered (if any) |
| `success` | bool | Whether the request succeeded |
| `tool_calls_count` | int | Number of tool calls in the request |
| `total_tool_bytes` | int | Aggregate bytes from tool interactions |
| `tool_latency_sec` | float | Total latency for tool execution |
| `is_streaming` | bool | Whether response was streamed |
| `chunk_count` | int | Number of chunks received (streaming) |
| `inter_chunk_times` | JSON | Array of inter-chunk time gaps in seconds |
| `network_profile` | string | Network conditions used |
| `metadata` | JSON | Extensible metadata (tool names, iteration, etc.) |

### Computed Properties

```python
ttft = t_first_token - t_request_start  # Time to First Token
ttlt = t_last_token - t_request_start   # Time to Last Token
```

---

## Metric Categories

### 1. Latency Metrics

Measures the time from request submission to response completion.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `latency_mean` | `mean(latencies)` | seconds | Average latency across all requests |
| `latency_median` | `median(latencies)` | seconds | 50th percentile latency |
| `latency_p50` | `percentile(latencies, 50)` | seconds | 50th percentile (linear interpolation; equals `latency_median`) |
| `latency_min` | `min(latencies)` | seconds | Minimum observed latency |
| `latency_max` | `max(latencies)` | seconds | Maximum observed latency |
| `latency_std` | `stdev(latencies)` | seconds | Standard deviation (0 if n=1) |
| `latency_p95` | `percentile(latencies, 95)` | seconds | 95th percentile latency |
| `latency_p99` | `percentile(latencies, 99)` | seconds | 99th percentile latency |

**Source:** `latency_sec` field from each `LogRecord`

**No-data sentinel:** the `latency_*`, traffic-volume (`*_bytes_mean`, `total_bytes_mean`) and `ul_dl_ratio_mean` fields use `null` when no qualifying record exists (truthy-`latency_sec` filter, §2.3); `0.0` is **not** a valid "no data" marker for them. JSON/3GPP report outputs coerce `null → 0` at serialization (`or 0`), preserving legacy schema shape.

---

### 2. Time to First Token (TTFT)

Critical QoE metric measuring perceived responsiveness for streaming responses.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `ttft_mean` | `mean(ttft_values)` | seconds | Average time to first token |
| `ttft_median` | `median(ttft_values)` | seconds | Median TTFT |
| `ttft_p50` | `percentile(ttft_values, 50)` | seconds | 50th percentile (same as median) |
| `ttft_p95` | `percentile(ttft_values, 95)` | seconds | 95th percentile TTFT |
| `ttft_p99` | `percentile(ttft_values, 99)` | seconds | 99th percentile TTFT |

**Calculation:**
```python
for each record:
    if t_first_token and t_request_start are set:
        ttft = t_first_token - t_request_start
        if ttft >= 0:
             ttft_values.append(ttft)
```

#### 2.1 TTFT Field Semantics (encoder_placement scenarios)

Applies to the whole `encoder_placement` family, including the `qwen36` client variants (identical client structure). Several TTFT-like fields coexist in the records; they are **not interchangeable**:

| Field | Anchor (t=0) | Covers | Status |
|-------|--------------|--------|--------|
| `LogRecord.t_first_token − t_request_start` (→ `metadata.ttft_sec`, `ttft_values`) | `t_request_start` captured by the scenario/worker | Full path: video read + edge preprocessing + ViT + upload + network + server prefill + first token | **Authoritative** — use for all aggregation and analysis |
| `metadata.e2e_ttft` (client-side) | client function entry `t_e2e_start` | Same as above within <1 ms (a few lines of function entry) | Redundant / historical; may drift by milliseconds — do **not** use for SQL reconciliation against `ttft_sec` |
| `engine_stats.server_ttft` (sidecar) | server `t_gen_start` (prefill submission; the preceding bundle `t_prepare` is reported separately) | Engine-side prefill + first token only; excludes upload, network, preprocessing | Server-perspective attribution only; **much smaller** than terminal-observed TTFT — never compare against terminal TTFT |
| `ScenarioResult.ttft_sec` (aggregation layer) | — | serial: first turn, single user; concurrent: first wave `ttft_p50` (see `metadata.ttft_semantics`) | Read `ttft_semantics` before comparing across scenarios |

**Comparison warnings:**

- **serial vs. concurrent TTFT is not directly comparable.** The concurrent protocol excludes model loading from the measurement window (workers are `ready` before the barrier), while a serial first request includes cold loading (mechanism and required filtering in §2.2 below).
- The reconcile lines in the client log (`[EncPl] reconcile:`) decompose the gap between `ttft_sec` and `server_ttft`. The expected gap is upload + network + preprocessing; an unexpectedly large gap usually means `e2e_ttft` was mistaken for `ttft_sec`, not a real network effect.

#### 2.2 Cold-Load Bias in Serial encoder_placement Runs

**Mechanism.** The serial `encoder_placement` scenario has no warmup phase, and the client's `AutoProcessor` and edge model (`_get_processor` / `_get_edge_model`) load lazily. The **first request of each session** therefore pays transformers import + weight loading *inside* the measurement window — tens of seconds on first use.

**Scope.** Serial scenarios only (including the `qwen36` serial variants, same structure): the first record of each session gets a cold-load-inflated `ttft_sec` / `latency_sec`. Concurrent scenarios are unaffected (loading happens in the ready phase before the barrier, outside the window), and all subsequent serial records hit process-level caches.

**Required analysis practice:**

- Multi-run serial comparisons: **discard the first record of each `run_index`** (or at minimum the first turn) before aggregating TTFT/latency.
- Serial `--runs 1`: TTFT/latency columns must not be used for cross-mode or cross-scenario comparison.
- Bench tools (`encpl_bench_eval*`): the serial tools (`encpl_bench_eval.py`, `encpl_bench_eval_qwen36.py`) warm the client caches before the question loop, so new runs have no cold first question; results files written **before** the warmup was added still contain one cold first entry (first question of the first video) — discard it when analyzing old files.

#### 2.3 Aggregation Caveats (encoder_placement)

- **Empty completions record latency as 0.0 — filtered by the analysis layer AND the wave stats; raw SQL does not.** When the model emits EOS on the first token (greedy decode, occasional prompts), the request is `ok=True` with `ttft = NULL` and `latency_sec = 0.0`. Both `analysis/metrics.py` (`if r.get("latency_sec")`) and the concurrent scenario/bench wave aggregations use truthiness filters, so `latency_mean` / `latency_p50` exclude the zeros; a raw `AVG(latency_sec)` in SQL **includes** them and pulls the mean down. When querying SQL directly, filter `tokens_out > 0` (or the concurrent wave summary's `wave_empty` count) yourself. Related: a record set consisting **only** of empty completions yields `null` `latency_*` / `*_bytes_mean` metrics (§1 sentinel note), not `0.0`.
- **`--users` above the sidecar's `--max-bundles` evicts in-flight bundles.** `BundleStore` evicts the oldest bundle when full; with all N users uploading before any POST consumes, N must stay ≤ `--max-bundles` (concurrent sidecar default 64) or a user's bundle can be LRU-evicted and fail with 404.
- **`ScenarioResult.total_latency_sec` is scenario-shaped.** Serial: sum of per-turn user latencies. Concurrent: sum of per-wave wall-clock (`wave_wall_sec`). The two are not comparable across scenario types — compare means or per-wave wall time instead.
- **Event timestamps are read-buffer quantized.** SSE events are parsed in batches at urllib3 read boundaries, so fast streams show several events sharing one timestamp; `inter_chunk_times` inherits this quantization. Treat inter-chunk jitter from the DB as coarse — the authoritative timing is pcap (`--capture-pcap`).
- **`_pct` (wave stats) uses linear interpolation — same semantics as `analysis/metrics.py _percentile`.** Historical note: before this unification it used round-to-index selection, under which an n=2 `p50` equaled the *minimum*; results produced before the fix interpret small-wave percentiles as order statistics. Current code interpolates (`k = (n−1)·p/100`, fractional part weighted between neighbors).
- **`wave_wall_sec` in edge-process mode includes process spawn, interpreter import, model load and teardown.** Every wave cold-starts its worker processes, so thread-mode and process-mode `wave_wall_sec` are not comparable. Use per-request fields (`ttft_sec`, `latency_sec`) or `wave_barrier_wall_sec` (barrier release → all workers joined; process mode only, emitted by both the concurrent scenarios and the concurrent bench tools) for cross-topology wave comparisons.
- **Dataset mode is playlist-shaped (qwen3/qwen36 concurrent scenarios): one barrier wave, per-user serial playlists.** The barrier aligns only each user's FIRST video; afterwards every user runs its playlist serially with no inter-video barrier (user u gets `items[u::N]` round-robin — same question→user assignment the old wave-chunking produced). Consequences: (1) `wave_wall_sec` spans the whole playlist marathon (a fast user finishing early does not wait for slow siblings), so N-user "wave" and per-request TTFT are still per-video comparable but `wave_wall_sec` no longer bounds a single synchronized request set; (2) `wave_ok`/`wave_empty`/`wave_videos` count VIDEOS, not users, and `ok_per_user`/`ttft_per_user`/`latency_per_user`/`video_paths_per_user` are nested per-video lists (`upload_bytes/time_per_user` are whole-playlist totals, `conns_per_user` is the first video's connections); (3) `turn_index = user_index*1000 + seq` (the old `wave*1000+user` layout is gone; rows carry explicit `seq` + `user_index`); (4) a failed video is one `ok=False` entry and the user's playlist CONTINUES — a user producing no record at all yields `ok_per_user[u] = None` in single-video mode or a `[]`/placeholder list in playlist mode; (5) `bench.padded` is always `false` in playlist mode (the final-wave filler duplicates are gone; the key stays for schema compatibility with older wave-chunked rows — aggregators skipping `padded=true` still work). Older rows from the wave-chunked era are identified by `wave_index > 0` in `waves` metadata / absent `seq` in record metadata.
- **`engine_stats.upload_time` is PUT send→response, not payload transfer time.** It includes server-side `torch.load`/dequantization of the bundle before the HTTP response is returned, so it overstates the pure wire-transfer time; use pcap for transfer-time attribution.
- **`server_ttft` carries a thread-hop of milliseconds (concurrent sidecar).** The first-token timestamp is taken when the uvicorn event loop dequeues the engine's output, so under concurrency the value includes engine-loop→uvicorn queueing jitter. Fine for attribution at the ~100 ms scale; do not treat single-digit-ms differences as engine-side signal.
- **Reconcile `edge_riders` is a residual, not a precise stage sum.** It is computed as `edge_prep_time − stage1 − stage2`, so it absorbs every unmodeled slice of `_prepare_edge_bundle` (riders, file reads, serialization) — read it as "unexplained edge prep", not as a named pipeline stage.
- **"overall TTFT p90" has two incompatible calibers.** The Qwen3 concurrent bench final summary (`encpl_bench_eval_concurrent.py` `_print_final_summary`) pools **raw per-request TTFT values** across waves and takes the p90 of that pool; the two runners (`run_encoder_placement_concurrent{,_qwen36}.py`) and the qwen36 bench instead take **the median of per-wave p90s** (`_pct(all_p90, 50)`). Both print the same "overall ... p90" label but answer different questions (request-tail vs typical-wave-tail); with few waves the median-of-p90s is markedly coarser. Never compare the two directly; for new analysis prefer the request-level pooling (recomputable row-by-row from the DB).
- **Percentile variants coexist in the chart/Excel layer.** `generate_charts.py` (nearest-rank `int(n·0.95)`; `np.percentile` elsewhere) and `export_to_excel.py` (`sorted[len//2]` upper-median) do not use the linear-interpolation semantics that `metrics.py`/`_pct` standardized. Values differ slightly for the same data; treat `metrics.py` output as the authoritative percentile caliber and never diff chart-vs-analysis numbers at small sample sizes.
- **chat / chat_token success-vs-failure rows use different anchors.** Success rows take the client's `t_request_start` (post message-serialization); failure rows take the scenario's pre-build anchor. The gap is request-build time (milliseconds). Failure rows are excluded from TTFT anyway; the asymmetry only matters for `latency_sec` on failure-heavy runs.

---

### 3. Time to Last Token (TTLT)

Measures total generation time for streaming responses.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `ttlt_mean` | `mean(ttlt_values)` | seconds | Average time to last token |
| `ttlt_median` | `median(ttlt_values)` | seconds | Median TTLT |
| `ttlt_p50` | `percentile(ttlt_values, 50)` | seconds | 50th percentile |
| `ttlt_p95` | `percentile(ttlt_values, 95)` | seconds | 95th percentile TTLT |
| `ttlt_p99` | `percentile(ttlt_values, 99)` | seconds | 99th percentile TTLT |

**Calculation:**
```python
for each record:
    if t_last_token and t_request_start are set:
        ttlt = t_last_token - t_request_start
        if ttlt >= 0:
            ttlt_values.append(ttlt)
```

---

### 4. Traffic Volume Metrics

Measures payload sizes.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `request_bytes_mean` | `mean(request_bytes)` | bytes | Average uplink payload size |
| `response_bytes_mean` | `mean(response_bytes)` | bytes | Average downlink payload size |
| `total_bytes_mean` | `request_bytes_mean + response_bytes_mean` | bytes | Average total traffic per request |
| `ul_dl_ratio_mean` | `request_bytes_mean / response_bytes_mean` | ratio | Uplink/downlink ratio |

**Byte Estimation Method:**
```python
def estimate_payload_bytes(obj):
    return len(json.dumps(obj, default=str).encode('utf-8'))
```

**SSE framing (encoder_placement clients).** For `encoder_placement` chat streams, `response_events[i].bytes` and `response_bytes` count the **full SSE line** — `data: ` prefix (6 B) + JSON payload + the `\n\n` event separator (2 B) — over **every** stream event: content chunks, usage chunks, role-only/finish_reason deltas, and the final `engine_stats` event. The invariant `sum(response_events[i].bytes) == response_bytes` holds per stream; check it in analysis to detect truncation. Caveat: `TRACE_MAX_EVENTS` / `TRACE_MAX_BYTES` truncate the event list written to the **trace JSON file** only (a `[trace truncated]` note marks it) — the DB's `response_bytes` is unaffected; run the invariant check on the bench results file, not the truncated trace. Data written before this convention was introduced is not directly comparable; check the capture date.

**`on_wire` estimates are coarse (encoder_placement upload path).** `metadata.*_wire_bytes` (and the helper `_on_wire_len`) estimate TCP framing as a fixed MSS = 1460 B with no ACK/Header overhead modeling. Two known distortions: (a) on loopback the MTU is 64 KB, so real segments are far larger than modeled; (b) delayed ACKs and Nagle interactions are not modeled. Treat wire numbers as **order-of-magnitude estimates only**; the authoritative byte counts on the cable come from `--capture-pcap` capture analysis. Note that since the `request_bytes` payload-semantics fix, `request_bytes` counts application-layer bytes only (upload payload + chat JSON); wire estimates remain available via `metadata.uploaded_wire_bytes` / `metadata.video_wire_bytes`.

---

### 5. Token Metrics

Measures LLM token consumption and generation rate.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `tokens_in_mean` | `mean(tokens_in)` | tokens | Average input tokens per request |
| `tokens_out_mean` | `mean(tokens_out)` | tokens | Average output tokens per request |
| `token_rate_mean` | `mean(tokens_out / latency_sec)` | tokens/sec | Average token generation rate |

**Token Rate Calculation:**
```python
token_rates = []
for record in records:
    if record.latency_sec > 0 and record.tokens_out:
        token_rates.append(record.tokens_out / record.latency_sec)
token_rate_mean = mean(token_rates)
```

**Rate-metric denominators are not interchangeable.** `token_rate_mean` (and ran2 `tokens_per_sec`) divide `tokens_out` by the **full `latency_sec`** (TTFT included), i.e. an *effective* end-to-end generation rate. `streaming_rate_mean` (§9) divides by `t_last_token − t_first_token` (TTFT excluded), i.e. the *decode-phase* streaming rate. The three coexisting denominators are intentional (different questions); never compare values across them — see §5/§9 boundary notes.

**Fallback Token Estimation:**
When the provider doesn't return token counts:
1. Use `tiktoken` library with `cl100k_base` encoding
2. Fall back to `len(text) // 4` approximation

**Fallback accuracy caveat:** `cl100k_base` is an OpenAI vocabulary; applied to Qwen tokenizer text it deviates systematically by ±20–30 %. In `encoder_placement` fallbacks (no usage chunk), `tokens_in` is estimated from the prompt alone (~20 tokens) while the true prefill includes the ~14.5k video tokens — treat fallback token counts as placeholders, never as prefill-volume evidence. Prefer the engine-reported usage counts whenever present.

---

### 6. Reliability Metrics

Measures service reliability and error rates.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `success_rate` | `(success_count / total_count) * 100` | % | Percentage of successful requests |
| `error_count` | `total_count - success_count` | count | Number of failed requests |

**Error Taxonomy:**
Categorizes failures by type:

| Error Type | Detection Criteria |
|------------|-------------------|
| `timeout` | `"timeout" in error_type` or `"timed out" in error_type` |
| `rate_limited` | `http_status == 429` or `"rate limit" in error_type` |
| `server_error` | `http_status >= 500` |
| `tool_failure` | Failed tool/computer action records |
| `other` | All other failures |

---

### 7. Agent/Tool Metrics

Measures AI agent behavior with external tool usage.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `tool_calls_mean` | `mean(tool_calls_count)` | calls | Average tool calls per request |
| `loop_factor` | `mean(records_per_session)` | factor | API calls per user prompt (agent iterations) |
| `tool_latency_mean` | `mean(tool_latency_sec)` | seconds | Average tool execution time |
| `tool_latency_p50` | `percentile(tool_latencies, 50)` | seconds | Median tool latency |
| `tool_latency_p95` | `percentile(tool_latencies, 95)` | seconds | 95th percentile tool latency |
| `tool_latency_p99` | `percentile(tool_latencies, 99)` | seconds | 99th percentile tool latency |

**Loop Factor Calculation:**
```python
# Approximates agent iterations by averaging session sizes
session_sizes = group_by_session(records).count()
loop_factor = mean(session_sizes)
```

---

### 8. Computer Use Metrics

Metrics specific to computer-use scenarios.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `computer_actions_mean` | `mean(actions_per_session)` | actions | Average GUI actions per session |
| `computer_action_latency_mean` | `mean(action_latencies)` | seconds | Average action execution time |
| `computer_action_latency_p95` | `percentile(action_latencies, 95)` | seconds | 95th percentile action latency |
| `computer_screenshot_bytes_mean` | `mean(screenshot_bytes)` | bytes | Average screenshot size |
| `computer_steps_mean` | `mean(steps_per_session)` | steps | Average steps per session |
| `computer_action_error_rate` | `errors / total_actions` | ratio | Action failure rate |

---

### 9. Streaming Metrics

Measures streaming response behavior and quality.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `chunk_count_mean` | `mean(chunk_count)` | chunks | Average chunks per streaming response |
| `streaming_rate_mean` | `mean(response_bytes / (t_last_token - t_first_token))` | bytes/sec | Average streaming throughput — note the duration **excludes TTFT** (first-token to last-token window) |

**Streaming Rate Calculation:**
```python
for record in streaming_records:
    duration = t_last_token - t_first_token
    if duration > 0:
        rate = response_bytes / duration
        streaming_rates.append(rate)
```

---

### 10. Streaming Stall Metrics

Detects and measures playback stalls in streaming responses.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `stall_event_count` | `count(gaps >= threshold)` | events | Number of stall events |
| `stall_rate` | `stall_count / total_gaps` | ratio | Proportion of gaps that are stalls |
| `stall_duration_mean` | `mean(stall_durations)` | seconds | Average stall duration |
| `stall_duration_p95` | `percentile(stall_durations, 95)` | seconds | 95th percentile stall duration |
| `stall_duration_p99` | `percentile(stall_durations, 99)` | seconds | 99th percentile stall duration |

**Stall Detection:**
```python
stall_gap_threshold_sec = 1.0  # configurable

for record in streaming_records:
    inter_chunk_times = json.loads(record.inter_chunk_times)
    for gap in inter_chunk_times:
        if gap >= stall_gap_threshold_sec:
            stall_durations.append(gap)
```

**Inter-Chunk Time Calculation:**
```python
# In StreamingResponse class
@property
def inter_chunk_times(self):
    times = []
    for i in range(1, len(self.chunks)):
        gap = self.chunks[i].timestamp - self.chunks[i-1].timestamp
        times.append(gap)
    return times
```

---

### 11. Burstiness Metrics

Characterizes traffic patterns for network planning.

| Metric | Formula | Unit | Description |
|--------|---------|------|-------------|
| `burst_peak_to_mean` | `max(total_bytes) / mean(total_bytes)` | ratio | Peak-to-mean traffic ratio |
| `burst_cv` | `stdev(total_bytes) / mean(total_bytes)` | ratio | Coefficient of variation |
| `burst_on_gap_count` | `count(gaps <= threshold)` | count | Number of "on" periods |
| `burst_off_gap_count` | `count(gaps > threshold)` | count | Number of "off" periods |
| `burst_on_gap_mean_sec` | `mean(on_gaps)` | seconds | Average duration of burst activity |
| `burst_off_gap_mean_sec` | `mean(off_gaps)` | seconds | Average duration of idle periods |

**Burstiness Calculation:**
```python
burst_gap_threshold_sec = 1.0  # configurable

# Calculate inter-request gaps
request_times = sorted([r.t_request_start for r in records])
gaps = [request_times[i] - request_times[i-1] for i in range(1, len(request_times))]

# Classify gaps as "on" (burst) or "off" (idle)
on_gaps = [g for g in gaps if g <= burst_gap_threshold_sec]
off_gaps = [g for g in gaps if g > burst_gap_threshold_sec]
```

---

## Calculation Methods

### Percentile Calculation

Uses linear interpolation between floor and ceiling values:

```python
def _percentile(values, p):
    """Calculate percentile using linear interpolation."""
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p / 100
    f = int(k)  # floor index
    c = f + 1 if f + 1 < len(sorted_values) else f  # ceiling index

    if f == c:
        return sorted_values[f]

    # Linear interpolation
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
```

### Record Filtering

The `MetricsCalculator` separates records by type:

```python
# Primary records: main API calls
primary_records = [r for r in records
    if r.metadata.get("type") not in ("mcp_tool_call", "computer_use_action", ...)]

# Tool records: MCP tool call results
tool_records = [r for r in records
    if r.metadata.get("type") == "mcp_tool_call"]

# Computer action records: GUI automation
computer_action_records = [r for r in records
    if r.metadata.get("type") == "computer_use_action"]
```

---

## Report Output Format

The `MetricsCalculator.to_3gpp_format()` method converts metrics to JSON format with millisecond precision.

### Output Structure

```json
{
  "scenario": "chat_streaming",
  "network_conditions": "5g_urban",
  "samples": 42,

  "qoe_metrics": {
    "time_to_first_token_ms": 245.3,
    "time_to_first_token_p50_ms": 230.1,
    "time_to_first_token_p95_ms": 412.7,
    "time_to_first_token_p99_ms": 523.4,
    "time_to_last_token_ms": 1523.6,
    "time_to_last_token_p50_ms": 1450.2,
    "time_to_last_token_p95_ms": 2134.8,
    "time_to_last_token_p99_ms": 2567.1,
    "response_latency_ms": 1523.6,
    "response_latency_p95_ms": 2134.8,
    "success_rate_pct": 98.5
  },

  "traffic_characteristics": {
    "uplink_bytes_avg": 1024,
    "downlink_bytes_avg": 4096,
    "ul_dl_ratio": 0.25,
    "token_rate_per_sec": 45.2
  },

  "ai_service_metrics": {
    "agent_loop_factor": 3.2,
    "tool_calls_avg": 2.1,
    "tool_latency_ms": 156.4,
    "tool_latency_p50_ms": 142.3,
    "tool_latency_p95_ms": 287.6,
    "tool_latency_p99_ms": 412.8
  },

  "computer_use_metrics": {
    "actions_avg": 5.3,
    "action_latency_ms": 234.5,
    "action_latency_p95_ms": 456.7,
    "screenshot_bytes_avg": 524288,
    "steps_avg": 3.2,
    "action_error_rate": 0.02
  },

  "streaming_metrics": {
    "stall_gap_threshold_ms": 1000,
    "stall_rate": 0.05,
    "stall_event_count": 3,
    "stall_duration_mean_ms": 1245.6,
    "stall_duration_p95_ms": 1876.3,
    "stall_duration_p99_ms": 2134.5
  },

  "error_taxonomy": {
    "timeout": 1,
    "rate_limited": 0,
    "server_error": 0,
    "tool_failure": 1,
    "other": 0
  },

  "burstiness": {
    "peak_to_mean": 3.45,
    "coefficient_of_variation": 0.78,
    "gap_threshold_ms": 1000,
    "on_gap_mean_ms": 234.5,
    "off_gap_mean_ms": 5678.9,
    "on_gap_count": 38,
    "off_gap_count": 4
  }
}
```

---

## Configuration Parameters

### Stall Detection Threshold

```yaml
# configs/scenarios.yaml
defaults:
  stall_gap_sec: 1.0  # Inter-chunk gap threshold for stall detection
```

Gaps between streaming chunks exceeding this threshold are classified as stalls.

### Burst Detection Threshold

```yaml
# configs/scenarios.yaml
defaults:
  burst_gap_sec: 1.0  # Inter-request gap threshold for burst analysis
```

Gaps between requests:
- `<= threshold`: "on" period (burst activity)
- `> threshold`: "off" period (idle)

---

## Complete Metric Summary

| Category | Count | Metrics |
|----------|-------|---------|
| Latency | 7 | mean, median, min, max, std, p95, p99 |
| TTFT | 5 | mean, median, p50, p95, p99 |
| TTLT | 5 | mean, median, p50, p95, p99 |
| Traffic | 4 | request_bytes, response_bytes, total_bytes, ul_dl_ratio |
| Tokens | 3 | tokens_in, tokens_out, token_rate |
| Reliability | 2 | success_rate, error_count |
| Agent/Tool | 6 | tool_calls, loop_factor, tool_latency (mean, p50, p95, p99) |
| Computer Use | 6 | actions, action_latency (mean, p95), screenshot_bytes, steps, error_rate |
| Streaming | 2 | chunk_count, streaming_rate |
| Stall | 5 | event_count, rate, duration (mean, p95, p99) |
| Burstiness | 6 | peak_to_mean, cv, on_gap (mean, count), off_gap (mean, count) |
| Error Types | 5 | timeout, rate_limited, server_error, tool_failure, other |
| **RAN2 Q1 — UL-heavy** | 8 | ul/dl bytes (totals + distributions), ul/dl ratio, per-dir pkt count/size, per-dir multi-window throughput |
| **RAN2 Q2 — Bursts** | 8 | per-dir bursts at 10/100ms, burst size/duration/peak-rate dists, inter-burst idle CDF, burstiness-by-window |
| **RAN2 Q3 — RTT** | 4 | TCP RTT, TLS handshake, HTTP setup-RTT, inter-chunk-gap vs RTT ratio, E2E-latency vs RTT |
| **RAN2 Q4 — Variability** | 6 | volume distributions, reliability-vs-loss, inter-burst-idle CV, flow-duration, connection-reuse, per-tool sub-flow |
| **RAN2 Q5 — Tokenized** | 4 | inter-token-gap dist per profile, tokens→bytes regression (UL+DL), token-rate vs DL-pkt-rate |
| **Subtotal RAN2** | **30** | See [RAN2 Methodology Metrics](#ran2-methodology-metrics-s4-260859) |
| **Total** | **86** | |

---

## RAN2 Methodology Metrics (S4-260859)

This section documents the metrics added in response to the RAN2 LS (S4-260703) per the methodology laid out in **S4-260859 Annex D**. Every metric is computed by `analysis/ran2_metrics.py::compute_ran2_metrics()` and consumed by the chart/report pipeline. The entry point is a single pure function:

```python
from analysis.ran2_metrics import compute_ran2_metrics

result = compute_ran2_metrics(
    records=db_records,               # list[dict] from traffic_logs table
    pcap_metrics=pcap_analyzer_output,  # optional list[PcapMetrics]
    profiles_yaml="configs/profiles.yaml",
)
# → {"generated_at": ..., "Q1": {...}, "Q2": {...}, "Q3": {...}, "Q4": {...}, "Q5": {...}}
```

The output is a nested dict keyed by RAN2 question. Each leaf distribution uses the shape `{n, min, p50, p95, p99, max, mean}` (from `_distribution()`), which matches the style of existing testbed metrics.

### Source data

| Source | Where | Used by |
|---|---|---|
| `traffic_logs` table (per-turn DB rows) | `logs/traffic_logs.db` | all five questions |
| `PacketRecord` list (per-packet pcap records) | `PcapMetrics.packets` (populated in `PcapAnalyzer.analyze()`) | Q1.3, Q1.4, Q2.1, Q2.2, Q2.3, Q3.1, Q3.2, Q4.5–Q4.7, Q5.3 |
| Per-flow TCP stats (handshake RTT, duration, retransmits) | `PcapMetrics.flows[*]` (`TCPFlow`) | Q3.1, Q3.2, Q4.6, Q4.7 |
| Streaming `inter_chunk_times` (DB column) | per-turn record | Q3.3, Q5.4 |
| `metadata.record_type == 'tool_call'` rows | per-turn record | Q4.7 |
| `configs/profiles.yaml:profiles[<name>].loss_pct` | file | Q4.4 |

### Q1 — UL-heavy traffic (`compute_ran2_metrics(...)["Q1"]`)

| # | Metric | Source | Formula / Notes | Field in output |
|---|---|---|---|---|
| Q1.1 | UL/DL byte volume per (scenario, profile) | DB `request_bytes`, `response_bytes` | Sum over successful primary turns; report total + distribution | `Q1.per_scenario_profile[<s/p>].ul_bytes_total / dl_bytes_total / ul_bytes_per_turn / dl_bytes_per_turn` |
| Q1.2 | UL/DL byte ratio | DB | `sum(request_bytes) / sum(response_bytes)` | `Q1.per_scenario_profile[<s/p>].ul_dl_ratio` |
| Q1.3 | Per-direction packet count + mean packet size | pcap `PacketRecord` | For each pcap: count pkts where `direction == ul/dl`; mean size = bytes/count | `Q1.pcap_per_direction[i].{ul_packets, dl_packets, ul_mean_pkt_size, dl_mean_pkt_size, ul_bytes_total, dl_bytes_total}` |
| Q1.4 | Per-direction throughput at 1/10/100 ms / 1 s / 10 s windows | pcap `PacketRecord.timestamp + size + direction` | Bucket packets by `int((ts-t0)/window)`; rate = `bucket_bytes · 8 / window` bps; emit `(rel_t, ul_bps, dl_bps)` tuples + peak Mbps per window | `Q1.pcap_per_direction[i].peak_mbps_by_window["1ms"|"10ms"|"100ms"|"1s"|"10s"]` plus full series on `PcapMetrics.throughput_by_window` |

**Code:** `analysis/pcap_analyzer.py::PcapAnalyzer._compute_per_direction_and_multi_window()` (post-processes `metrics.packets` once analyze() completes).

### Q2 — Data bursts and delay-bound (`Q2`)

| # | Metric | Source | Formula / Notes | Field in output |
|---|---|---|---|---|
| Q2.1 | Burst detection with **>10 ms / 100 ms** idle-gap per direction | pcap `PacketRecord` sorted per direction | Walk packets; when inter-arrival `iat > gap_sec`, close current burst and start a new one. Emit `{start, end, duration_sec, total_bytes, packet_count, peak_rate_bps}` per burst. | `Q2.per_pcap[i].burst_stats_by_gap["10ms"\|"100ms"][<ul/dl>]` (full bursts list on `PcapMetrics.bursts_by_gap`) |
| Q2.2 | Burst peak intra-burst rate + inter-burst idle CDF | same | `peak_rate_bps = total_bytes·8 / duration_sec` per burst. Inter-burst idle = `iat` when it crosses the gap threshold. | `Q2.per_pcap[i].burst_stats_by_gap[<gap>][<dir>].peak_rate_mbps` (distribution) and `Q2.per_pcap[i].interburst_idle_by_gap[<gap>][<dir>].cdf_sec` |
| Q2.3 | Burstiness index (peak/mean) at **1 ms / 10 ms / 100 ms / 1 s / 10 s** windows | pcap multi-window bucketing | For each window, sum UL+DL bytes per bucket, then `peak / mean` of those sums. | `Q2.per_pcap[i].burstiness_by_window["1ms"..."10s"]` |
| Q2.4 | TTFB (already supported) | DB `t_first_token - t_request_start` | `ttft_values` distribution | `Q2.per_scenario_profile_delay[<s/p>].ttft_sec` |
| Q2.5 | TTLB (already supported) | DB `t_last_token - t_request_start` | `ttlt_values` distribution | `Q2.per_scenario_profile_delay[<s/p>].ttlt_sec` |

**Code:** burst segmentation + idle gaps in `analysis/pcap_analyzer.py::_compute_per_direction_and_multi_window()` (uses `burst_gaps_sec=(0.010, 0.100)` by default). Distribution aggregation in `analysis/ran2_metrics.py::_q2_bursts()`.

**Caveat (per S4-260859 editor's note):** windows below ~10 ms may be affected by OS scheduler jitter; treat `burstiness_by_window["1ms"]` as indicative.

### Q3 — Round-trip delay (`Q3`)

| # | Metric | Source | Formula / Notes | Field in output |
|---|---|---|---|---|
| Q3.1 | TCP handshake RTT (min/median/p95) | pcap `TCPFlow.handshake_rtt` (`ack_time − syn_time`) | Collected during `_process_tcp_packet()`; aggregated into `{min, p50, p95, p99, max, mean}` distribution in ms. | `Q3.tcp_rtt` |
| Q3.2a | TLS handshake time | per-turn `metadata.tls.handshake_ms` | Aggregated across all primary turns. Honors both `handshake_ms` and `handshake_sec` keys. | `Q3.tls_handshake` |
| Q3.2b | HTTP connection setup RTT | pcap `TCPFlow.time_to_first_data − TCPFlow.handshake_duration` | Time between the end of TCP handshake and the first application data byte. Clamped at 0. | `Q3.http_setup_rtt` |
| Q3.3 | Inter-chunk gap vs RTT (streaming) | DB `inter_chunk_times[]` + `Q3.tcp_rtt.p50` | `ratio_p50 = median(inter_chunk_sec) · 1000 / rtt_p50_ms`. Quantifies how much of a stream's inter-chunk gap is attributable to network RTT vs server-side generation. | `Q3.inter_chunk_vs_rtt.{inter_chunk_sec, tcp_rtt_p50_ms, ratio_p50}` |
| Q3.4 | E2E response latency vs RTT (non-streaming) | DB `latency_sec` + `Q3.tcp_rtt.p50` | Per-turn ratio `(latency·1000)/rtt_p50_ms` aggregated per `(scenario, profile)`. | `Q3.e2e_latency_vs_rtt[<s/p>]` |

**Code:** `analysis/ran2_metrics.py::_q3_rtt()`. TCP handshake times come from `_process_tcp_packet()` in `pcap_analyzer.py`.

### Q4 — Intra-application variability (`Q4`)

| # | Metric | Source | Formula / Notes | Field in output |
|---|---|---|---|---|
| Q4.1 | Volume distributions (min/p50/p95/max) + packet-count distribution | DB `request_bytes`, `response_bytes`; pcap `TCPFlow.packets_sent + packets_recv` | Per-scenario: byte distributions from DB, packet-count-per-flow distribution aggregated across all pcap flows in scope. | `Q4.volume_distribution[<scenario>].{request_bytes, response_bytes, packet_count_per_flow}` |
| Q4.2 | Per-burst size/duration distributions | reuse Q2 burst output | Same `_distribution(sizes)` and `_distribution(durs)` per gap label, per direction. | `Q2.per_pcap[i].burst_stats_by_gap[<gap>][<dir>].{size_bytes, duration_sec}` |
| Q4.3 | Per-turn TTFB/TTLB across netem profiles | DB | Already computed per (scenario, profile); render as overlay in charts. | `Q2.per_scenario_profile_delay[<s/p>]` |
| Q4.4 | Reliability vs netem loss rate | DB success flag + profile `loss_pct` from `configs/profiles.yaml` | `success_rate = ok / total` per (scenario, profile). Paired with the profile's nominal `loss_pct`. | `Q4.reliability_by_loss_pct[<s/p>].{turns, success, success_rate, profile_loss_pct}` |
| Q4.5 | CV of inter-burst idle time per direction | pcap inter-burst idle lists | `cv = stdev/mean`. Returns None on <2 samples or zero mean. | `Q4.inter_arrival_cv[<pcap>][<gap>][<direction>]` |
| Q4.6 | Flow duration distribution; flows per pcap; connection-reuse ratio | pcap `TCPFlow` | Duration from `TCPFlow.duration`; reuse = `flow_keys_seen_more_than_once / total_flows`; flows-per-pcap = unique flow_keys per pcap. | `Q4.connection_duration.{flow_duration_sec, flows_per_pcap, connection_reuse_ratio}` |
| Q4.7 | Distinct destination IPs/ports per pcap; per-tool sub-flow volume | pcap `TCPFlow.dst_ip + dst_port`; DB `metadata.record_type=='tool_call'` + `metadata.tool_name` | Distinct `(dst_ip, dst_port)` tuples per pcap; for each tool name: total `request_bytes`, `response_bytes`, `tool_latency_sec`. | `Q4.agentic_flows.{distinct_dests_per_pcap, per_tool_bytes}` |

**Code:** `analysis/ran2_metrics.py::_q4_variability()` and `_per_tool_bytes()`.

**Formulas worth noting explicitly:**

```python
# Q4.5 — Coefficient of Variation of inter-burst idle gaps
def _cv(values):
    if len(values) < 2: return None
    mean = statistics.mean(values)
    if mean == 0: return None
    return statistics.stdev(values) / mean

# Q4.6 — Connection reuse ratio
# A flow_key is (src_ip:src_port-dst_ip:dst_port). Same 4-tuple seen
# more than once across pcaps ⇒ connection reused (e.g. HTTP/2 or
# keep-alive), not a new connection per turn.
reuse_hits    = sum(1 for fk in flow_keys if already_seen[fk])
reuse_total   = total_flows
reuse_ratio   = reuse_hits / reuse_total
```

### Q5 — Tokenized traffic (`Q5`)

| # | Metric | Source | Formula / Notes | Field in output |
|---|---|---|---|---|
| Q5.1 | Token counts per turn (distributions) | DB `tokens_in`, `tokens_out` | Provider-reported when available, `tiktoken`-estimated fallback. | `Q5.token_counts_by_scenario[<scenario>].{tokens_in, tokens_out}` |
| Q5.2 | Token rate (tokens/s) per scenario | DB `tokens_out / latency_sec` | Per-turn rate; distribution across turns. | `Q5.token_counts_by_scenario[<scenario>].tokens_per_sec` |
| Q5.3 | Token-arrival rate vs DL-pkt-arrival rate | DB `inter_chunk_times[]`; pcap DL packets | Token rate per profile = `1 / p50(inter_chunk_sec)`; DL pkt rate per pcap = `len(dl_pkts) / (last_ts − first_ts)`. | `Q5.token_arrival_vs_pkt_arrival.{token_rate_per_profile_hz, dl_pkt_rate_hz}` |
| Q5.4 | Inter-token gap distribution per network profile | DB `inter_chunk_times[]` (streaming only) | Aggregate across streaming turns per profile → `{n, min, p50, p95, p99, max, mean}` in seconds. | `Q5.inter_token_gap_by_profile[<profile>]` |
| Q5.5 | Tokens → UL/DL byte regression | DB `(tokens_in, request_bytes)`, `(tokens_out, response_bytes)` | Least-squares: `slope = cov(x,y)/var(x)`; `intercept = ȳ − slope·x̄`; `r² = 1 − ss_res/var_y`. | `Q5.token_to_bytes_regression_by_scenario[<scenario>].{ul, dl}` each `{n, slope, intercept, r2}` |
| Q5.6 | Error resilience / traffic for real-time tokens | (deferred) | Per the pCR, real-time token format still under SA4 study; scaffolding in place, no metric emitted yet. | — |

**Code:** `analysis/ran2_metrics.py::_q5_tokenized()` and `_least_squares()`.

**Interpretation of Q5.5 (the piece RAN2 asked for explicitly):**

The regression coefficients let RAN2 translate a **token-count traffic model** into a **byte-count traffic model**:

- `bytes_ul ≈ slope_ul · tokens_in + intercept_ul`
- `bytes_dl ≈ slope_dl · tokens_out + intercept_dl`

where `slope` is bytes per token (~1.3 × 4 bytes/token for ASCII-heavy English text, plus HTTP framing overhead absorbed into the intercept). `r²` indicates how well the linear model fits — near 1.0 means a clean linear relationship that RAN2 can use directly; lower values imply the relationship is multi-modal or dominated by fixed-size headers (e.g. base64 attachments dwarfing the token payload in multimodal scenarios).

### Helper: distribution shape

Every RAN2 distribution in the output uses this shape, produced by `_distribution()`:

```python
def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "min": min(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
        "mean": statistics.mean(values),
    }
```

`_percentile()` uses linear interpolation between the two nearest ranks (matches NumPy's default `linear` method).

### Enabling the RAN2 metric pipeline

The metrics compute regardless of whether pcap is captured — pcap-dependent fields return empty distributions when `pcap_metrics` is empty. To produce the complete set:

```bash
# Capture pcap during the run so Q1.3/Q1.4/Q2.x/Q3.1/Q3.2/Q4.5/Q4.6/Q4.7 populate:
./run_full_tests.sh                   # CAPTURE_PCAP defaults to true

# Regenerate RAN2 metrics (and feed them into charts/report):
python generate_charts.py --pcap-dir results/captures
python generate_results_md.py
```

When pcap is disabled (`CAPTURE_PCAP=false`), the DB-only metrics (Q1.1, Q1.2, Q2.4, Q2.5, Q3.3, Q3.4, Q4.1 partial, Q4.3, Q4.4, Q5.1, Q5.2, Q5.4, Q5.5) still emit; the pcap-only ones are skipped with `n=0` distributions.
