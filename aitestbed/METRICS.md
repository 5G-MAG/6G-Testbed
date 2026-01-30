# METRICS.md

Detailed documentation of all metrics calculated by the 6G AI Traffic Characterization Testbed (ATCT).

## Table of Contents

1. [Overview](#overview)
2. [Data Capture Layer](#data-capture-layer)
3. [Metric Categories](#metric-categories)
4. [Calculation Methods](#calculation-methods)
5. [3GPP Output Format](#3gpp-output-format)
6. [Configuration Parameters](#configuration-parameters)

---

## Overview

The testbed calculates metrics at multiple levels:

1. **Per-Request Level** - Raw measurements captured during each API call (`LogRecord`)
2. **Aggregated Level** - Statistical summaries per scenario/profile combination (`ScenarioMetrics`)
3. **3GPP Report Level** - Formatted output aligned with 3GPP SA4 6G Media Study requirements

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
| `latency_min` | `min(latencies)` | seconds | Minimum observed latency |
| `latency_max` | `max(latencies)` | seconds | Maximum observed latency |
| `latency_std` | `stdev(latencies)` | seconds | Standard deviation (0 if n=1) |
| `latency_p95` | `percentile(latencies, 95)` | seconds | 95th percentile latency |
| `latency_p99` | `percentile(latencies, 99)` | seconds | 99th percentile latency |

**Source:** `latency_sec` field from each `LogRecord`

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

Measures payload sizes for network capacity planning (3GPP Section b).

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

**Fallback Token Estimation:**
When the provider doesn't return token counts:
1. Use `tiktoken` library with `cl100k_base` encoding
2. Fall back to `len(text) // 4` approximation

---

### 6. Reliability Metrics (QoE - 3GPP Section a)

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

### 7. Agent/Tool Metrics (3GPP Section d)

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

Metrics specific to computer-use scenarios (GUI automation).

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
| `streaming_rate_mean` | `mean(response_bytes / duration)` | bytes/sec | Average streaming throughput |

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

## 3GPP Output Format

The `MetricsCalculator.to_3gpp_format()` method converts metrics to 3GPP-aligned JSON format with millisecond precision.

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

### Unit Conversions

| Internal Unit | Output Unit | Conversion |
|--------------|-------------|------------|
| seconds | milliseconds | `value * 1000` |
| ratio | ratio | unchanged |
| bytes | bytes | unchanged |
| count | count | unchanged |
| percent | percent | unchanged |

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

## Source Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| LogRecord | `analysis/logger.py` | 16-94 |
| TrafficLogger | `analysis/logger.py` | 96-200 |
| MetricsCalculator | `analysis/metrics.py` | 113-436 |
| Percentile function | `analysis/metrics.py` | 439-450 |
| 3GPP format | `analysis/metrics.py` | 495-559 |
| ChatResponse | `clients/base.py` | 129-146 |
| StreamingResponse | `clients/base.py` | 57-127 |
| BaseScenario._create_log_record | `scenarios/base.py` | 111-183 |

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
| **Total** | **56** | |
