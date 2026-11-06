#!/bin/bash
# =============================================================================
# 6G AI Traffic Testbed - Run Script
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
IMAGE="6g-ai-testbed:latest"
NETWORK_EMULATION=""
INTERACTIVE=""
VOLUMES="-v $(pwd)/logs:/app/logs -v $(pwd)/results/reports:/app/results/reports -v $(pwd)/results/captures:/app/results/captures -v $(pwd)/results/l7_captures:/app/results/l7_captures"
ENV_FILE=""
EXTRA_ARGS=""

# Load .env file if it exists
if [[ -f ".env" ]]; then
    ENV_FILE="--env-file .env"
fi

# Help message
show_help() {
    echo ""
    echo -e "${BLUE}6G AI Traffic Characterization Testbed${NC}"
    echo "========================================"
    echo ""
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Options:"
    echo "  -i, --interactive    Run in interactive mode"
    echo "  -n, --netem          Enable network emulation (requires root)"
    echo "  --no-volumes         Don't mount local volumes"
    echo "  --image IMAGE        Use specific image (default: 6g-ai-testbed:latest)"
    echo "  --help               Show this help message"
    echo ""
    echo "Commands:"
    echo "  --help               Show testbed help"
    echo "  --shell              Start interactive shell"
    echo "  --list-scenarios     List available scenarios"
    echo "  --list-profiles      List available network profiles"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Show testbed help"
    echo "  $0 --help"
    echo ""
    echo "  # List available scenarios"
    echo "  $0 --list-scenarios"
    echo ""
    echo "  # Run a chat scenario"
    echo "  $0 python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 5"
    echo ""
    echo "  # Run shopping agent with network emulation"
    echo "  $0 -n python orchestrator.py --scenario shopping_agent --profile cell_edge --runs 10"
    echo ""
    echo "  # Interactive shell"
    echo "  $0 -i --shell"
    echo ""
    echo "  # Run full test matrix"
    echo "  $0 python orchestrator.py --scenario all --runs 10"
    echo ""
    echo "Environment:"
    echo "  API keys are loaded from .env file (copy from .env.example)"
    echo "  Results are saved to ./logs and ./results/reports directories"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interactive)
            INTERACTIVE="-it"
            shift
            ;;
        -n|--netem)
            NETWORK_EMULATION="--cap-add=NET_ADMIN -e ENABLE_NETEM=true"
            shift
            ;;
        --no-volumes)
            VOLUMES=""
            shift
            ;;
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --help)
            if [[ $# -eq 1 ]]; then
                show_help
                exit 0
            else
                # Pass --help to the container
                break
            fi
            ;;
        *)
            # Remaining arguments are passed to the container
            break
            ;;
    esac
done

# If no command provided, show help
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

# Build command
CMD="docker run --rm"
CMD="$CMD $INTERACTIVE"
CMD="$CMD $NETWORK_EMULATION"
CMD="$CMD $VOLUMES"
CMD="$CMD $ENV_FILE"
CMD="$CMD $IMAGE"
CMD="$CMD $@"

# Show what we're running
echo -e "${BLUE}Running:${NC} $CMD"
echo ""

# Execute
exec $CMD
