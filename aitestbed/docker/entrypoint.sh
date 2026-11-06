#!/bin/bash
set -e

# =============================================================================
# 6G AI Traffic Testbed - Docker Entrypoint
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Environment validation
# =============================================================================
validate_env() {
    log_info "Validating environment..."

    local missing_vars=()

    # Check for required API keys based on scenario
    if [[ "$1" == *"openai"* ]] || [[ "$1" == *"chat"* ]] || [[ "$1" == *"image"* ]]; then
        if [[ -z "$OPENAI_API_KEY" ]]; then
            missing_vars+=("OPENAI_API_KEY")
        fi
    fi

    if [[ "$1" == *"gemini"* ]]; then
        if [[ -z "$GOOGLE_API_KEY" ]]; then
            missing_vars+=("GOOGLE_API_KEY")
        fi
    fi

    if [[ "$1" == *"agent"* ]] || [[ "$1" == *"search"* ]]; then
        if [[ -z "$BRAVE_API_KEY" ]]; then
            log_warn "BRAVE_API_KEY not set - web search scenarios may fail"
        fi
    fi

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        echo "Set them using: docker run -e VAR=value ..."
        exit 1
    fi

    log_success "Environment validated"
}

# =============================================================================
# Network emulation setup
# =============================================================================
setup_network() {
    if [[ "$ENABLE_NETEM" == "true" ]]; then
        log_info "Setting up network emulation..."

        # Check if we have NET_ADMIN capability
        if ! sudo tc qdisc show >/dev/null 2>&1; then
            log_warn "Network emulation requires --cap-add=NET_ADMIN"
            log_warn "Continuing without network emulation..."
            export ENABLE_NETEM=false
            return
        fi

        log_success "Network emulation available"
    fi
}

# =============================================================================
# MCP servers verification
# =============================================================================
verify_mcp_servers() {
    log_info "Verifying MCP servers..."

    # Check npm-based MCP servers
    local npm_servers=(
        "@modelcontextprotocol/server-brave-search"
        "@modelcontextprotocol/server-filesystem"
        "@modelcontextprotocol/server-memory"
    )

    for server in "${npm_servers[@]}"; do
        if npx --yes "$server" --version >/dev/null 2>&1; then
            log_success "  npm: $server"
        else
            log_warn "  npm: $server (not available)"
        fi
    done

    # Check Python-based MCP servers
    if python -c "import mcp_server_fetch" >/dev/null 2>&1; then
        log_success "  python: mcp-server-fetch"
    else
        log_warn "  python: mcp-server-fetch (not installed)"
    fi

    log_success "MCP server check complete"
}

# =============================================================================
# Main command handling
# =============================================================================
case "$1" in
    --help|-h)
        echo ""
        echo "6G AI Traffic Characterization Testbed"
        echo "======================================"
        echo ""
        echo "Usage: docker run [options] testbed <command>"
        echo ""
        echo "Commands:"
        echo "  --list-scenarios     List available test scenarios"
        echo "  --list-profiles      List available network profiles"
        echo "  --run-scenario       Run a specific scenario"
        echo "  --run-matrix         Run the full test matrix"
        echo "  --shell              Start interactive shell"
        echo "  --help               Show this help message"
        echo ""
        echo "Examples:"
        echo "  # List scenarios"
        echo "  docker run testbed --list-scenarios"
        echo ""
        echo "  # Run chat scenario"
        echo "  docker run -e OPENAI_API_KEY=sk-... testbed \\"
        echo "    python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 5"
        echo ""
        echo "  # Run with network emulation (requires NET_ADMIN)"
        echo "  docker run --cap-add=NET_ADMIN -e ENABLE_NETEM=true \\"
        echo "    -e OPENAI_API_KEY=sk-... testbed \\"
        echo "    python orchestrator.py --scenario shopping_agent --profile cell_edge"
        echo ""
        echo "  # Interactive shell"
        echo "  docker run -it -e OPENAI_API_KEY=sk-... testbed --shell"
        echo ""
        echo "  # Mount volumes for persistent data"
        echo "  docker run -v ./logs:/app/logs -v ./reports:/app/reports \\"
        echo "    -e OPENAI_API_KEY=sk-... testbed \\"
        echo "    python orchestrator.py --scenario all --runs 10"
        echo ""
        echo "Environment Variables:"
        echo "  OPENAI_API_KEY       OpenAI API key (required for OpenAI scenarios)"
        echo "  GOOGLE_API_KEY       Google API key (required for Gemini scenarios)"
        echo "  BRAVE_API_KEY        Brave Search API key (for agent scenarios)"
        echo "  ENABLE_NETEM         Enable network emulation (true/false)"
        echo "  LOG_LEVEL            Logging level (DEBUG, INFO, WARNING, ERROR)"
        echo ""
        exit 0
        ;;

    --list-scenarios)
        python orchestrator.py --list-scenarios
        ;;

    --list-profiles)
        python orchestrator.py --list-profiles
        ;;

    --shell)
        log_info "Starting interactive shell..."
        exec /bin/bash
        ;;

    python|python3)
        validate_env "$*"
        setup_network
        verify_mcp_servers
        log_info "Executing: $*"
        exec "$@"
        ;;

    *)
        # Default: pass through to orchestrator or execute command
        if [[ -f "$1" ]] || [[ "$1" == *.py ]]; then
            validate_env "$*"
            setup_network
            verify_mcp_servers
            exec python "$@"
        else
            validate_env "$*"
            setup_network
            exec "$@"
        fi
        ;;
esac
