# TRACES

Run-level traces and sample request/response payloads.

## Selection Notes

- Data source: `logs/traffic_logs.db`
- Selection: latest fully successful session per scenario (fallback to latest session)
- Trace files are generated only when `TRACE_PAYLOADS=1` is set during runs.

## SDP Offer/Answer Samples (WebRTC)

Latest offer/answer pair captured during realtime WebRTC sessions for Realtime Audio WebRTC from `logs/sdp/`.

**Offer:** `logs/sdp/20260125_105241_e6e6b998_offer.sdp` (3138 bytes)
```sdp
v=0
o=- 3978348757 3978348757 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic:WMS *
m=audio 48208 UDP/TLS/RTP/SAVPF 96 0 8
c=IN IP4 10.255.255.254
a=sendrecv
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 urn:ietf:params:rtp-hdrext:ssrc-audio-level
a=mid:0
a=msid:4dda2902-0ad7-4af0-ba8a-dca711689779 32a8c48a-1c76-478c-b990-2d3d83af6e54
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc:2770269086 cname:22a5eb72-890c-468c-a89c-1ee54337bede
a=rtpmap:96 opus/48000/2
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=candidate:04ca8b06d45d14b04e47bfd6f27e21ea 1 udp 2130706431 10.255.255.254 48208 typ host
a=candidate:5e8a9423a15ffcaef2a35218a505bc95 1 udp 2130706431 192.168.10.65 47442 typ host
a=candidate:d58b7930e644a07acbe48891f509c72c 1 udp 2130706431 fdc8:d1ab:b2a6:0:129e:4e2c:b75c:c847 47595 typ host
a=candidate:d6fd1cf191e588e4f0d6b6e60956c1d4 1 udp 2130706431 fdc8:d1ab:b2a6::f3b 46816 typ host
a=candidate:1d98a1972b49d82afa64940b682d2694 1 udp 2130706431 fdc8:d1ab:b2a6:0:69cd:c5e2:a42a:9f5 45413 typ host
a=candidate:04c6d03278c06af4d7d47141f5463284 1 udp 1694498815 76.233.245.125 47442 typ srflx raddr 192.168.10.65 rport 47442
a=end-of-candidates
a=ice-ufrag:lWpC
a=ice-pwd:I4b4bkUznzNS30UbuJFVul
a=fingerprint:sha-256 89:52:B7:7B:38:BD:02:0F:8B:B0:7D:62:C5:9A:4F:28:D8:3C:76:28:01:DE:F1:21:BA:C5:5F:EA:31:67:C7:80
a=fingerprint:sha-384 DA:1D:7F:CC:B9:EA:A7:69:F7:F3:E6:61:3E:C2:A2:3F:7F:95:09:6B:5D:20:31:B7:76:3C:E8:14:82:25:FA:92:F0:7A:97:AB:9C:EA:6F:C8:3D:E4:F2:E1:38:94:17:BC
a=fingerprint:sha-512 19:73:D5:CD:7A:FF:63:C3:69:57:E8:20:70:56:D5:FC:3D:86:9A:BA:62:DE:45:3C:80:39:17:44:DC:0A:0D:B8:F3:9E:75:6B:7C:9F:2B:D0:05:99:13:7F:A9:5F:29:F7:FC:15:C1:A8:6D:F6:3A:08:2F:BA:6A:90:08:0A:49:0B
a=setup:actpass
m=application 45901 DTLS/SCTP 5000
c=IN IP4 10.255.255.254
a=mid:1
a=sctpmap:5000 webrtc-datachannel 65535
a=max-message-size:65536
a=candidate:04ca8b06d45d14b04e47bfd6f27e21ea 1 udp 2130706431 10.255.255.254 45901 typ host
a=candidate:5e8a9423a15ffcaef2a35218a505bc95 1 udp 2130706431 192.168.10.65 45229 typ host
a=candidate:d58b7930e644a07acbe48891f509c72c 1 udp 2130706431 fdc8:d1ab:b2a6:0:129e:4e2c:b75c:c847 44921 typ host
a=candidate:d6fd1cf191e588e4f0d6b6e60956c1d4 1 udp 2130706431 fdc8:d1ab:b2a6::f3b 45517 typ host
a=candidate:1d98a1972b49d82afa64940b682d2694 1 udp 2130706431 fdc8:d1ab:b2a6:0:69cd:c5e2:a42a:9f5 47070 typ host
a=candidate:04c6d03278c06af4d7d47141f5463284 1 udp 1694498815 76.233.245.125 45229 typ srflx raddr 192.168.10.65 rport 45229
a=end-of-candidates
a=ice-ufrag:yiVJ
a=ice-pwd:XRCtnyQOjJJdCvUu5B1f7o
a=fingerprint:sha-256 89:52:B7:7B:38:BD:02:0F:8B:B0:7D:62:C5:9A:4F:28:D8:3C:76:28:01:DE:F1:21:BA:C5:5F:EA:31:67:C7:80
a=fingerprint:sha-384 DA:1D:7F:CC:B9:EA:A7:69:F7:F3:E6:61:3E:C2:A2:3F:7F:95:09:6B:5D:20:31:B7:76:3C:E8:14:82:25:FA:92:F0:7A:97:AB:9C:EA:6F:C8:3D:E4:F2:E1:38:94:17:BC
a=fingerprint:sha-512 19:73:D5:CD:7A:FF:63:C3:69:57:E8:20:70:56:D5:FC:3D:86:9A:BA:62:DE:45:3C:80:39:17:44:DC:0A:0D:B8:F3:9E:75:6B:7C:9F:2B:D0:05:99:13:7F:A9:5F:29:F7:FC:15:C1:A8:6D:F6:3A:08:2F:BA:6A:90:08:0A:49:0B
a=setup:actpass
```

**Answer:** `logs/sdp/20260125_105241_e6e6b998_answer.sdp` (1483 bytes)
```sdp
v=0
o=- 7327372874735784009 1769359961 IN IP4 0.0.0.0
s=-
t=0 0
a=msid-semantic:WMS*
a=fingerprint:sha-256 AB:18:19:08:F7:E6:7C:07:25:D6:A5:F8:68:3D:19:AC:E1:2F:23:3B:57:AB:7B:35:52:A6:4A:C2:6B:2C:42:C7
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 96 0 8
c=IN IP4 0.0.0.0
a=setup:active
a=mid:0
a=ice-ufrag:KYOFLyRb/u1
a=ice-pwd:qfMWH7hmL5yVK1CGNTCa39H0DHOj2TsJ
a=rtcp-mux
a=rtcp-rsize
a=rtpmap:96 opus/48000/2
a=fmtp:96 minptime=10;useinbandfec=1
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=ssrc:569148451 cname:realtimeapi
a=ssrc:569148451 msid:realtimeapi audio
a=ssrc:569148451 mslabel:realtimeapi
a=ssrc:569148451 label:audio
a=msid:realtimeapi audio
a=sendrecv
a=candidate:727169150 1 udp 2130706431 4.151.200.38 3478 typ host ufrag KYOFLyRb/u1
a=candidate:1878291698 1 tcp 1671430143 4.151.200.38 443 typ host tcptype passive ufrag KYOFLyRb/u1
a=candidate:38269317 1 udp 2130706431 172.214.226.198 3478 typ host ufrag KYOFLyRb/u1
a=candidate:2394539241 1 tcp 1671430143 172.214.226.198 443 typ host tcptype passive ufrag KYOFLyRb/u1
a=candidate:4152413238 1 udp 2130706431 172.203.39.49 3478 typ host ufrag KYOFLyRb/u1
a=candidate:1788861106 1 tcp 1671430143 172.203.39.49 443 typ host tcptype passive ufrag KYOFLyRb/u1
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 0.0.0.0
a=setup:active
a=mid:1
a=sendrecv
a=sctp-port:5000
a=max-message-size:1073741823
a=ice-ufrag:KYOFLyRb/u1
a=ice-pwd:qfMWH7hmL5yVK1CGNTCa39H0DHOj2TsJ
```

## Computer Control - Provider A

### Run Metadata

| Field | Value |
|---|---|
| Scenario ID | `computer_control_agent` |
| Session ID | `4522d3dd-175f-4d8e-889e-10dc86ee97fe` |
| Provider | Provider A |
| Model | Model X |
| Network Profile | `5qi_80` |
| Run Index | 0 |
| Start | 2026-01-28 11:42:10 |
| End | 2026-01-28 11:44:33 |
| Turns | 39 |
| Streaming | false |
| Success Rate | 100.0% |

### Prompt Set

- Prompt 1: Open https://example.com, capture the page title and main headings, and save a short report to /tmp/testbed-sandbox/computer_control_report.md.
- Prompt 2: Search for 'OpenAI tools computer use' in the browser and summarize the first result.

### Sample Request (exact payload)

```json
{
  "format": "openai.responses.create",
  "payload": {
    "model": "computer-use-preview",
    "tools": [
      {
        "type": "computer_use_preview",
        "display_width": 1280,
        "display_height": 720,
        "environment": "browser"
      }
    ],
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Open https://example.com, capture the page title and main headings, and save a short report to /tmp/testbed-sandbox/computer_control_report.md."
          }
        ]
      }
    ],
    "truncation": "auto"
  }
}
```

### Sample Response (exact payload)

```json
{
  "format": "openai.responses.response",
  "payload": {
    "id": "resp_056d930d2351953f00697a4a714dac8196896ccce9ad65f6bf",
    "created_at": 1769622129.0,
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "metadata": {},
    "model": "computer-use-preview-2025-03-11",
    "object": "response",
    "output": [
      {
        "id": "cu_056d930d2351953f00697a4a7242788196b3a3c9e97862bf39",
        "action": {
          "type": "screenshot"
        },
        "call_id": "call_5pk2pVrJSLTu5xyYgsWupKbt",
        "pending_safety_checks": [],
        "status": "completed",
        "type": "computer_call"
      }
    ],
    "parallel_tool_calls": true,
    "temperature": 1.0,
    "tool_choice": "auto",
    "tools": [
      {
        "display_height": 720,
        "display_width": 1280,
        "environment": "browser",
        "type": "computer_use_preview"
      }
    ],
    "top_p": 1.0,
    "background": false,
    "completed_at": 1769622130.0,
    "conversation": null,
    "max_output_tokens": null,
    "max_tool_calls": null,
    "previous_response_id": null,
    "prompt": null,
    "prompt_cache_key": null,
    "prompt_cache_retention": null,
    "reasoning": {
      "effort": "medium",
      "generate_summary": null,
      "summary": null
    },
    "safety_identifier": null,
    "service_tier": "default",
    "status": "completed",
    "text": {
      "format": {
        "type": "text"
      },
      "verbosity": "medium"
    },
    "top_logprobs": 0,
    "truncation": "auto",
    "usage": {
      "input_tokens": 523,
      "input_tokens_details": {
        "cached_tokens": 0
      },
      "output_tokens": 7,
      "output_tokens_details": {
        "reasoning_tokens": 0
      },
      "total_tokens": 530
    },
    "user": null,
    "billing": {
      "payer": "developer"
    },
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "store": true
  }
}
```

### Throughput Sample (per second)

- Throughput file: `logs/traces/throughput_computer_control_agent_4522d3dd.json`
- Throughput plot: `logs/traces/figures/throughput_computer_control_agent_4522d3dd.png`
- Unit: Kbps

```json
[
  {
    "second": 0,
    "timestamp": 1769622129.0,
    "offset_sec": 0,
    "request_bytes": 393.0,
    "response_bytes": 968.1626609523984,
    "ul_kbps": 3.144,
    "dl_kbps": 7.745301287619187
  },
  {
    "second": 1,
    "timestamp": 1769622130.0,
    "offset_sec": 1,
    "request_bytes": 6120.0,
    "response_bytes": 4812.696299505064,
    "ul_kbps": 48.96,
    "dl_kbps": 38.50157039604051
  },
  {
    "second": 2,
    "timestamp": 1769622131.0,
    "offset_sec": 2,
    "request_bytes": 0.0,
    "response_bytes": 473.0662670526415,
    "ul_kbps": 0.0,
    "dl_kbps": 3.784530136421132
  },
  {
    "second": 3,
    "timestamp": 1769622132.0,
    "offset_sec": 3,
    "request_bytes": 0.0,
    "response_bytes": 473.0662670526415,
    "ul_kbps": 0.0,
    "dl_kbps": 3.784530136421132
  },
  {
    "second": 4,
    "timestamp": 1769622133.0,
    "offset_sec": 4,
    "request_bytes": 0.0,
    "response_bytes": 464.0085054372546,
    "ul_kbps": 0.0,
    "dl_kbps": 3.7120680434980367
  }
]
```

### Throughput Plot (per run)

![Per-Second Throughput](logs/traces/figures/throughput_computer_control_agent_4522d3dd.png)

### Turn Trace

| Turn | Success | Latency (s) | Tokens In | Tokens Out | Request Bytes | Response Bytes | TTFT (s) | TTLT (s) | Chunks | Trace File |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | yes | 1.486 | 523 | 7 | 393 | 1444 | - | - | 0 | `logs/traces/20260128_114210_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 0.289 | - | - | 22 | 4253 | - | - | 0 | - |
| 0 | yes | 3.158 | 1378 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114213_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 1.273 | - | - | 16 | 4253 | - | - | 0 | - |
| 0 | yes | 4.564 | 2237 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114219_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 1.268 | - | - | 16 | 4253 | - | - | 0 | - |
| 0 | yes | 5.732 | 3096 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114226_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 1.270 | - | - | 16 | 4253 | - | - | 0 | - |
| 0 | yes | 7.611 | 3994 | 48 | 6098 | 1661 | - | - | 0 | `logs/traces/20260128_114235_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 0.273 | - | - | 22 | 4253 | - | - | 0 | - |
| 0 | yes | 6.636 | 4853 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114242_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 1.276 | - | - | 16 | 4253 | - | - | 0 | - |
| 0 | yes | 4.496 | 5712 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114248_computer_control_agent_4522d3dd_turn000.json` |
| 0 | yes | 1.277 | - | - | 16 | 4253 | - | - | 0 | - |
| 0 | yes | 4.976 | 5731 | 32 | 6098 | 1441 | - | - | 0 | `logs/traces/20260128_114254_computer_control_agent_4522d3dd_turn000.json` |
| 1 | yes | 1.501 | 509 | 7 | 335 | 1444 | - | - | 0 | `logs/traces/20260128_114256_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 0.270 | - | - | 22 | 4253 | - | - | 0 | - |
| 1 | yes | 4.507 | 1364 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114301_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.270 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 5.065 | 2223 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114307_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.272 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 4.604 | 3082 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114313_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 3.684 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 6.315 | 3941 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114323_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.270 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 5.714 | 4800 | 11 | 6098 | 1500 | - | - | 0 | `logs/traces/20260128_114330_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 0.273 | - | - | 22 | 4253 | - | - | 0 | - |
| 1 | yes | 12.300 | 5679 | 29 | 6098 | 1655 | - | - | 0 | `logs/traces/20260128_114343_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.271 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 8.011 | 5698 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114352_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.272 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 11.957 | 5746 | 50 | 6098 | 1692 | - | - | 0 | `logs/traces/20260128_114405_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 0.289 | - | - | 53 | 4253 | - | - | 0 | - |
| 1 | yes | 5.271 | 5777 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114411_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.273 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 5.457 | 5796 | 11 | 6098 | 1494 | - | - | 0 | `logs/traces/20260128_114418_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 1.270 | - | - | 16 | 4253 | - | - | 0 | - |
| 1 | yes | 13.508 | 5839 | 45 | 6098 | 1692 | - | - | 0 | `logs/traces/20260128_114432_computer_control_agent_4522d3dd_turn001.json` |
| 1 | yes | 0.277 | - | - | 53 | 4253 | - | - | 0 | - |
