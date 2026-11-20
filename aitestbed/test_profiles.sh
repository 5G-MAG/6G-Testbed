#!/bin/bash
#
# Matrix Smoke Validation Script
# Runs each scenario in configs/scenarios.yaml:test_matrix once per requested
# profile to catch scenario/runtime errors before full evaluation.
#
# By default exercises a representative spread covering: a clean baseline
# (no_emulation), a typical symmetric cellular case (5g_urban), an asymmetric
# satellite case that exercises the uplink: parser (satellite_leo), and a
# stress case (congested). Override with VALIDATION_PROFILES=a,b,c,...
#

set -u
set -o pipefail

if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

# Profile list to validate. Comma-separated. Default covers:
#   no_emulation  — baseline / parser-only
#   5g_urban      — symmetric reference
#   satellite_leo — asymmetric (exercises uplink: merge)
#   congested     — heavy stress (original smoke default)
VALIDATION_PROFILES=${VALIDATION_PROFILES:-no_emulation,5g_urban,satellite_leo,congested}

# Backwards-compatible single-profile shortcut. If set, overrides the list.
if [[ -n "${VALIDATION_PROFILE:-}" ]]; then
    VALIDATION_PROFILES="$VALIDATION_PROFILE"
fi

RUNS_PER_SCENARIO=${RUNS_PER_SCENARIO:-1}
NETWORK_INTERFACE=${NETWORK_INTERFACE:-auto}
INTER_SCENARIO_DELAY=${INTER_SCENARIO_DELAY:-1}
CAPTURE_PCAP=${CAPTURE_PCAP:-false}
CAPTURE_DIR=${CAPTURE_DIR:-results/captures_smoke}
CAPTURE_FILTER=${CAPTURE_FILTER:-"port 443 or port 80 or port 8080 or port 8000"}
CAPTURE_LOOPBACK=${CAPTURE_LOOPBACK:-true}  # Secondary tcpdump on lo for MCP-over-HTTP frames
MCP_TRANSPORT=${MCP_TRANSPORT:-http}  # MCP server transport: http (default, netem-shaped) or stdio
INCLUDE_OPTIONAL_SCENARIOS=${INCLUDE_OPTIONAL_SCENARIOS:-false}
RUN_TIMEOUT_SEC=${RUN_TIMEOUT_SEC:-600}
SCENARIO_FILTER=${SCENARIO_FILTER:-}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0
FAILURES=()
FAILURE_DETAILS=()
FAILURE_LOGS=()

HAS_GOOGLE_API=false
HAS_PLAYWRIGHT=false
HAS_MULTIMODAL_IMAGES=false
HAS_ALPACA=false
HAS_VLLM=false
LAST_INTERFACE=""

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_pass() { echo -e "  ${GREEN}PASS${NC}  $1"; ((PASS++)); }
log_fail() {
    local scenario=$1
    local detail=${2:-"unknown error"}
    local logfile=${3:-""}
    echo -e "  ${RED}FAIL${NC}  ${scenario} — ${detail}"
    ((FAIL++))
    FAILURES+=("$scenario")
    FAILURE_DETAILS+=("$detail")
    FAILURE_LOGS+=("$logfile")
}
log_skip() { echo -e "  ${YELLOW}SKIP${NC}  $1"; ((SKIP++)); }
log_phase() { echo -e "\n${BLUE}────────────────────────────────────────${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}────────────────────────────────────────${NC}"; }

check_sudo() {
    log_info "Checking sudo access for network emulation..."
    if ! sudo -n tc qdisc show dev lo &>/dev/null; then
        echo -e "${RED}ERROR: Cannot run tc commands without a password.${NC}"
        echo "Configure passwordless sudo for tc, or run 'sudo -v' in your terminal first."
        exit 1
    fi
    log_info "sudo access OK"
}

cleanup_network_state() {
    local interface=${1:-}
    local ifb_device=${IFB_DEVICE:-ifb0}

    if [[ -n "$interface" && "$interface" != "auto" ]]; then
        sudo tc qdisc del dev "$interface" root >/dev/null 2>&1 || true
        sudo tc qdisc del dev "$interface" ingress >/dev/null 2>&1 || true
    fi

    # Loopback and IFB cleanup are safe no-ops when nothing is configured.
    sudo tc qdisc del dev lo root >/dev/null 2>&1 || true
    sudo tc qdisc del dev lo ingress >/dev/null 2>&1 || true
    sudo tc qdisc del dev "$ifb_device" root >/dev/null 2>&1 || true
    sudo ip link set dev "$ifb_device" down >/dev/null 2>&1 || true
}

cleanup_on_exit() {
    cleanup_network_state "${LAST_INTERFACE:-}"
}

print_usage() {
    cat <<EOF
Usage: $0 [--scenario NAME] [--profile NAME[,NAME...]] [--asymmetric-only]
          [--video-only] [--help]

Options:
  --scenario NAME      Run only the named test_matrix scenario
  --profile LIST       Comma-separated profile list to validate (overrides
                       VALIDATION_PROFILES). Each scenario runs once per profile.
  --asymmetric-only    Shortcut for --profile satellite_leo,satellite_geo
                       (verifies the uplink: parser + apply_settings path)
  --video-only         Shortcut for --scenario video_understanding_vllm
  --help               Show this help text

Environment overrides:
  VALIDATION_PROFILES        Comma-separated profile list
                             (default: no_emulation,5g_urban,satellite_leo,congested)
  VALIDATION_PROFILE         Single-profile shortcut (overrides VALIDATION_PROFILES)
  RUNS_PER_SCENARIO          Runs per scenario per profile (default: 1)
  NETWORK_INTERFACE          tc interface override (default: auto)
  INTER_SCENARIO_DELAY       Delay between runs in seconds (default: 1)
  CAPTURE_PCAP               Enable pcap capture (default: false)
  CAPTURE_DIR                Capture output directory
  CAPTURE_FILTER             tcpdump filter
  INCLUDE_OPTIONAL_SCENARIOS Include matrix entries disabled by default
  RUN_TIMEOUT_SEC            Per-run timeout in seconds (default: 600, 0 = no timeout)
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)
                if [[ $# -lt 2 || -z "$2" ]]; then
                    echo -e "${RED}ERROR: --scenario requires a scenario name.${NC}"
                    exit 1
                fi
                SCENARIO_FILTER="$2"
                shift 2
                ;;
            --profile)
                if [[ $# -lt 2 || -z "$2" ]]; then
                    echo -e "${RED}ERROR: --profile requires a profile name (comma-separated for multiple).${NC}"
                    exit 1
                fi
                VALIDATION_PROFILES="$2"
                shift 2
                ;;
            --asymmetric-only)
                VALIDATION_PROFILES="satellite_leo,satellite_geo"
                shift
                ;;
            --video-only)
                SCENARIO_FILTER="video_understanding_vllm"
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}ERROR: Unknown option: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
}

check_optional_prereqs() {
    log_info "Checking optional prerequisites..."

    if [[ -n "${GOOGLE_SEARCH_API_KEY:-}" && -n "${GOOGLE_SEARCH_CX:-}" ]]; then
        HAS_GOOGLE_API=true
        log_info "  + Google Search API configured"
    else
        log_warn "  - Google Search API not configured"
    fi

    if python -c "import playwright" 2>/dev/null; then
        if python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(headless=True); b.close(); p.stop()" 2>/dev/null; then
            HAS_PLAYWRIGHT=true
            log_info "  + Playwright with Chromium ready"
        else
            log_warn "  - Playwright installed but Chromium missing"
        fi
    else
        log_warn "  - Playwright not installed"
    fi

    if [[ -f "examples/assets/network_diagram.png" ]]; then
        HAS_MULTIMODAL_IMAGES=true
        log_info "  + Multimodal test images found"
    else
        log_warn "  - Multimodal test images not found"
    fi

    if [[ -n "${ALPACA_API_KEY:-}" && -n "${ALPACA_SECRET_KEY:-}" ]]; then
        HAS_ALPACA=true
        log_info "  + Alpaca Market Data API configured"
    else
        log_warn "  - Alpaca not configured"
    fi

    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        HAS_VLLM=true
        log_info "  + vLLM server reachable at localhost:8000"
    else
        log_warn "  - vLLM server not reachable"
    fi

    echo ""
}

load_matrix_entries() {
    mapfile -t MATRIX_ENTRIES < <(
        python - <<'PY'
import yaml
from pathlib import Path

config = yaml.safe_load(Path("configs/scenarios.yaml").read_text())
scenarios = config.get("scenarios") or {}
matrix = config.get("test_matrix") or []

for entry in matrix:
    scenario_name = entry["scenario"]
    scenario_def = scenarios.get(scenario_name, {})
    profiles = entry.get("profiles") or []
    fields = [
        entry.get("phase", ""),
        scenario_name,
        "true" if entry.get("runner_enabled_by_default", True) else "false",
        "true" if scenario_def.get("disabled", False) else "false",
        ",".join(profiles),
        ",".join(entry.get("prereqs", [])),
    ]
    print("\t".join(fields))
PY
    )

    if [[ ${#MATRIX_ENTRIES[@]} -eq 0 ]]; then
        echo -e "${RED}ERROR: No test_matrix entries found in configs/scenarios.yaml.${NC}"
        exit 1
    fi
}

# Set REQUESTED_PROFILES from VALIDATION_PROFILES, validating each name
# exists in configs/profiles.yaml. Fails fast on typos.
load_requested_profiles() {
    REQUESTED_PROFILES=()
    IFS=',' read -r -a REQUESTED_PROFILES <<< "$VALIDATION_PROFILES"

    if [[ ${#REQUESTED_PROFILES[@]} -eq 0 ]]; then
        echo -e "${RED}ERROR: VALIDATION_PROFILES is empty.${NC}"
        exit 1
    fi

    local known
    known=$(python - <<'PY'
from netemu import load_profile_catalog
p = load_profile_catalog("configs/profiles.yaml")
print(" ".join(p["profiles"]))
PY
    )
    local missing=()
    for prof in "${REQUESTED_PROFILES[@]}"; do
        prof="${prof// /}"  # strip whitespace
        if ! [[ " $known " == *" $prof "* ]]; then
            missing+=("$prof")
        fi
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}ERROR: profile(s) not defined in configs/profiles.yaml: ${missing[*]}${NC}"
        exit 1
    fi
}

ensure_selected_scenario_exists() {
    local found=false
    local entry=""
    local scenario=""

    [[ -z "$SCENARIO_FILTER" ]] && return 0

    for entry in "${MATRIX_ENTRIES[@]}"; do
        IFS=$'\t' read -r _phase scenario _runner_enabled _scenario_disabled _profiles_csv _prereqs_csv <<< "$entry"
        if [[ "$scenario" == "$SCENARIO_FILTER" ]]; then
            found=true
            break
        fi
    done

    if [[ "$found" != "true" ]]; then
        echo -e "${RED}ERROR: Scenario '$SCENARIO_FILTER' is not present in configs/scenarios.yaml:test_matrix.${NC}"
        exit 1
    fi
}

# Intersection helper: print profiles that are both requested AND included
# in the entry's profiles list. Order follows REQUESTED_PROFILES.
profiles_for_entry() {
    local entry_profiles_csv=$1
    local -a entry_profiles
    IFS=',' read -r -a entry_profiles <<< "$entry_profiles_csv"
    local out=()
    local req prof
    for req in "${REQUESTED_PROFILES[@]}"; do
        req="${req// /}"
        for prof in "${entry_profiles[@]}"; do
            if [[ "$prof" == "$req" ]]; then
                out+=("$req")
                break
            fi
        done
    done
    (IFS=,; echo "${out[*]:-}")
}

get_scenario_config_value() {
    local scenario=$1
    local field=$2
    python - <<PY 2>/dev/null
import yaml
from pathlib import Path

config = yaml.safe_load(Path("configs/scenarios.yaml").read_text())
scenario = (config.get("scenarios") or {}).get("$scenario", {})
value = scenario.get("$field")
if value is None:
    raise SystemExit(1)
print(value)
PY
}

prereq_satisfied() {
    local prereq=$1

    case "$prereq" in
        ALPACA_API_KEY|ALPACA_SECRET_KEY)
            [[ "$HAS_ALPACA" == "true" ]]
            ;;
        GOOGLE_SEARCH_API_KEY|GOOGLE_SEARCH_CX)
            [[ "$HAS_GOOGLE_API" == "true" ]]
            ;;
        playwright_chromium)
            [[ "$HAS_PLAYWRIGHT" == "true" ]]
            ;;
        examples/assets/network_diagram.png)
            [[ "$HAS_MULTIMODAL_IMAGES" == "true" ]]
            ;;
        http://localhost:8000/v1/models)
            [[ "$HAS_VLLM" == "true" ]]
            ;;
        http://*|https://*)
            curl -sf "$prereq" > /dev/null 2>&1
            ;;
        */*|.*)
            [[ -e "$prereq" ]]
            ;;
        *)
            [[ -n "${!prereq:-}" ]]
            ;;
    esac
}

entry_skip_reason() {
    local runner_enabled=$1
    local scenario_disabled=$2
    local applicable_profiles_csv=$3
    local prereqs_csv=$4
    local missing=()
    local prereq=""

    if [[ "$runner_enabled" != "true" && "$INCLUDE_OPTIONAL_SCENARIOS" != "true" ]]; then
        echo "optional matrix entry skipped by default (set INCLUDE_OPTIONAL_SCENARIOS=true to include it)"
        return 0
    fi

    if [[ "$scenario_disabled" == "true" ]]; then
        echo "scenario is disabled in configs/scenarios.yaml"
        return 0
    fi

    if [[ -z "$applicable_profiles_csv" ]]; then
        echo "test_matrix entry includes none of the requested profiles ($VALIDATION_PROFILES)"
        return 0
    fi

    IFS=',' read -r -a prereqs <<< "$prereqs_csv"
    for prereq in "${prereqs[@]}"; do
        [[ -z "$prereq" ]] && continue
        if ! prereq_satisfied "$prereq"; then
            missing+=("$prereq")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        printf 'missing prereqs: %s' "${missing[*]}"
        return 0
    fi

    return 1
}

run_smoke_scenario() {
    local scenario=$1
    local profile=$2
    local scenario_interface="$NETWORK_INTERFACE"
    local interface_override=""
    local loopback_interface=""
    local logfile="logs/smoke_${scenario}_${profile}.log"
    local -a cmd=(
        python orchestrator.py
        --scenario "$scenario"
        --profile "$profile"
        --runs "$RUNS_PER_SCENARIO"
        --interface "$scenario_interface"
        --mcp-transport "$MCP_TRANSPORT"
    )
    if [[ "$NETWORK_INTERFACE" == "auto" ]]; then
        interface_override=$(get_scenario_config_value "$scenario" "network_interface" || true)
        if [[ -n "$interface_override" ]]; then
            scenario_interface="$interface_override"
            cmd=(
                python orchestrator.py
                --scenario "$scenario"
                --profile "$profile"
                --runs "$RUNS_PER_SCENARIO"
                --interface "$scenario_interface"
                --mcp-transport "$MCP_TRANSPORT"
            )
        fi
    fi

    loopback_interface=$(get_scenario_config_value "$scenario" "loopback_interface" || true)

    if [[ "$CAPTURE_PCAP" == "true" ]]; then
        cmd+=(
            --capture-pcap
            --capture-dir "$CAPTURE_DIR"
        )
        if [[ -n "$CAPTURE_FILTER" ]]; then
            cmd+=(
                --capture-filter "$CAPTURE_FILTER"
            )
        fi
        # Secondary loopback capture so MCP JSON-RPC frames (which traverse
        # http://127.0.0.1 between agent and MCP server) are visible in pcap.
        # Skipped automatically by orchestrator.py when the primary interface
        # is already lo or when --mcp-transport=stdio.
        if [[ "$CAPTURE_LOOPBACK" == "true" ]]; then
            cmd+=( --capture-loopback )
        else
            cmd+=( --no-capture-loopback )
        fi
    fi

    log_info "Running ${scenario} with profile ${profile} (${RUNS_PER_SCENARIO} run, interface: ${scenario_interface})"
    if [[ -n "$loopback_interface" ]]; then
        log_info "  Scenario-managed loopback shaping: ${loopback_interface}"
    fi

    LAST_INTERFACE="$scenario_interface"
    cleanup_network_state "$scenario_interface"
    LAST_LOGFILE="$logfile"

    if [[ "$RUN_TIMEOUT_SEC" -gt 0 ]]; then
        cmd+=(--run-timeout "$RUN_TIMEOUT_SEC")
    fi

    cmd+=(--stop-on-error)

    "${cmd[@]}" 2>&1 | tee "$logfile"
    rc=${PIPESTATUS[0]}
    cleanup_network_state "$scenario_interface"
    return "$rc"
}

extract_failure_summary() {
    local logfile=$1
    python - <<'PY' "$logfile"
from pathlib import Path
import re
import sys

logfile = Path(sys.argv[1])
try:
    lines = logfile.read_text(errors="ignore").splitlines()
except Exception:
    print("unable to read failure log")
    raise SystemExit(0)

patterns = [
    r"\bERROR\b",
    r"Completed: success=False",
    r"Traceback",
    r"Exception",
    r"RuntimeError",
    r"ValueError",
    r"TypeError",
    r"AssertionError",
    r"failed",
    r"timed out",
    r"server_error",
    r"rate_limited",
    r"tool_failure",
    r"Errors:",
    r"Retry \d+/\d+ after .*\(",
    r"\b429\b",
    r"\b5\d\d\b",
]

matches = []
for line in lines:
    text = line.strip()
    if not text:
        continue
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        if text not in matches:
            matches.append(text)

if matches:
    print(" | ".join(matches[-3:])[:800])
elif lines:
    tail = [line.strip() for line in lines[-5:] if line.strip()]
    print(" | ".join(tail)[:800] if tail else "non-zero exit without a recognizable error line")
else:
    print("non-zero exit without log output")
PY
}

scenario_reported_failure() {
    local logfile=$1
    python - <<'PY' "$logfile"
from pathlib import Path
import re
import sys

logfile = Path(sys.argv[1])
try:
    lines = logfile.read_text(errors="ignore").splitlines()
except Exception:
    raise SystemExit(1)

completed_lines = [line for line in lines if "Completed: success=" in line]
if completed_lines:
    last = completed_lines[-1]
    match = re.search(r"Completed: success=(True|False)", last)
    if match and match.group(1) == "False":
        raise SystemExit(0)

error_lines = [line for line in lines if " - ERROR - " in line]
if error_lines:
    raise SystemExit(0)

raise SystemExit(1)
PY
}

parse_args "$@"

echo "========================================"
echo "Test Matrix Smoke Validation"
echo "========================================"
echo "Profiles:          $VALIDATION_PROFILES"
echo "Runs per scenario: $RUNS_PER_SCENARIO  (per profile)"
echo "Packet capture:    $CAPTURE_PCAP"
echo "Optional entries:  $INCLUDE_OPTIONAL_SCENARIOS"
echo "Run timeout:       ${RUN_TIMEOUT_SEC}s"
if [[ -n "$SCENARIO_FILTER" ]]; then
    echo "Scenario filter:   $SCENARIO_FILTER"
fi
echo ""

mkdir -p logs
if [[ "$CAPTURE_PCAP" == "true" ]]; then
    mkdir -p "$CAPTURE_DIR"
fi
trap cleanup_on_exit EXIT INT TERM

check_sudo
check_optional_prereqs
load_matrix_entries
load_requested_profiles
ensure_selected_scenario_exists

CURRENT_PHASE=""
for entry in "${MATRIX_ENTRIES[@]}"; do
    IFS=$'\t' read -r phase scenario runner_enabled scenario_disabled profiles_csv prereqs_csv <<< "$entry"

    if [[ -n "$SCENARIO_FILTER" && "$scenario" != "$SCENARIO_FILTER" ]]; then
        continue
    fi

    if [[ "$phase" != "$CURRENT_PHASE" ]]; then
        CURRENT_PHASE="$phase"
        log_phase "$phase"
    fi

    applicable_csv=$(profiles_for_entry "$profiles_csv")

    skip_reason=$(entry_skip_reason "$runner_enabled" "$scenario_disabled" "$applicable_csv" "$prereqs_csv" || true)
    if [[ -n "$skip_reason" ]]; then
        log_skip "$scenario — $skip_reason"
        echo ""
        continue
    fi

    IFS=',' read -r -a applicable_profiles <<< "$applicable_csv"
    for prof in "${applicable_profiles[@]}"; do
        if run_smoke_scenario "$scenario" "$prof"; then
            if scenario_reported_failure "$LAST_LOGFILE"; then
                error_summary=$(extract_failure_summary "$LAST_LOGFILE")
                log_fail "${scenario}/${prof}" "$error_summary" "$LAST_LOGFILE"
            else
                log_pass "${scenario}/${prof}"
            fi
        else
            error_summary=$(extract_failure_summary "$LAST_LOGFILE")
            log_fail "${scenario}/${prof}" "$error_summary" "$LAST_LOGFILE"
        fi

        echo ""
        sleep "$INTER_SCENARIO_DELAY"
    done
done

echo "========================================"
echo -e "Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${SKIP} skipped${NC}"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed scenarios:${NC}"
    for i in "${!FAILURES[@]}"; do
        echo "  - ${FAILURES[$i]}: ${FAILURE_DETAILS[$i]}"
        if [[ -n "${FAILURE_LOGS[$i]}" ]]; then
            echo "    log: ${FAILURE_LOGS[$i]}"
        fi
    done
    exit 1
fi

echo -e "${GREEN}Smoke validation completed without scenario failures.${NC}"
