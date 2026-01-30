# =============================================================================
# 6G AI Traffic Characterization Testbed - Makefile
# =============================================================================

.PHONY: help build run shell test clean logs report \
        compose-up compose-down compose-logs

# Default target
help:
	@echo ""
	@echo "6G AI Traffic Characterization Testbed"
	@echo "======================================="
	@echo ""
	@echo "Docker Commands:"
	@echo "  make build          Build the Docker image"
	@echo "  make build-nocache  Build without cache"
	@echo "  make run            Run with default settings"
	@echo "  make shell          Start interactive shell"
	@echo "  make test           Run a quick test scenario"
	@echo ""
	@echo "Docker Compose Commands:"
	@echo "  make compose-up     Start testbed service"
	@echo "  make compose-down   Stop services"
	@echo "  make compose-logs   View service logs"
	@echo ""
	@echo "Scenario Commands:"
	@echo "  make chat           Run chat scenario"
	@echo "  make agent          Run shopping agent scenario"
	@echo "  make search         Run web search agent scenario"
	@echo "  make image          Run image generation scenario"
	@echo "  make realtime       Run real-time conversation scenario"
	@echo "  make all            Run all scenarios"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make logs           Show recent logs"
	@echo "  make report         Generate analysis report"
	@echo "  make clean          Remove containers and volumes"
	@echo "  make clean-all      Remove everything including images"
	@echo ""
	@echo "Setup:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Add your API keys to .env"
	@echo "  3. Run 'make build' to build the image"
	@echo "  4. Run 'make test' to verify setup"
	@echo ""

# =============================================================================
# Docker Commands
# =============================================================================

build:
	@chmod +x docker/*.sh
	docker build -t 6g-ai-testbed:latest .

build-nocache:
	docker build --no-cache -t 6g-ai-testbed:latest .

run:
	./docker/run.sh --help

shell:
	./docker/run.sh -i --shell

test:
	@echo "Running quick test (chat scenario, 2 runs)..."
	./docker/run.sh python orchestrator.py --scenario chat_basic --profile ideal_6g --runs 2

# =============================================================================
# Docker Compose Commands
# =============================================================================

compose-up:
	docker compose up -d testbed

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

# =============================================================================
# Scenario Commands
# =============================================================================

chat:
	./docker/run.sh python orchestrator.py --scenario chat_basic --profile 5g_urban --runs 10

chat-stream:
	./docker/run.sh python orchestrator.py --scenario chat_streaming --profile 5g_urban --runs 10

agent:
	./docker/run.sh python orchestrator.py --scenario shopping_agent --profile 5g_urban --runs 10

search:
	./docker/run.sh python orchestrator.py --scenario web_search_agent --profile 5g_urban --runs 10

image:
	./docker/run.sh python orchestrator.py --scenario image_generation --profile ideal_6g --runs 5

all:
	./docker/run.sh python orchestrator.py --scenario all --runs 10

# Direct web search scenarios (no MCP)
direct-search:
	./docker/run.sh python orchestrator.py --scenario direct_web_search --profile 5g_urban --runs 10

direct-search-burst:
	./docker/run.sh python orchestrator.py --scenario direct_web_search_burst --profile ideal_6g --runs 5

search-benchmark:
	./docker/run.sh python orchestrator.py --scenario parallel_search_benchmark --profile ideal_6g --runs 3

# Real-time conversational AI scenarios
realtime:
	./docker/run.sh python orchestrator.py --scenario realtime_text --profile 5g_urban --runs 5

realtime-interactive:
	./docker/run.sh python orchestrator.py --scenario realtime_interactive --profile 5g_urban --runs 5

realtime-technical:
	./docker/run.sh python orchestrator.py --scenario realtime_technical --profile 5g_urban --runs 5

realtime-audio:
	./docker/run.sh python orchestrator.py --scenario realtime_audio --profile ideal_6g --runs 5

# Run with network emulation
chat-netem:
	./docker/run.sh -n python orchestrator.py --scenario chat_basic --profile cell_edge --runs 10

agent-netem:
	./docker/run.sh -n python orchestrator.py --scenario shopping_agent --profile satellite --runs 10

direct-search-netem:
	./docker/run.sh -n python orchestrator.py --scenario direct_web_search --profile cell_edge --runs 10

# =============================================================================
# Utility Commands
# =============================================================================

logs:
	@echo "Recent experiment logs:"
	@ls -la logs/*.db 2>/dev/null || echo "No logs found"
	@echo ""
	@echo "Recent reports:"
	@ls -la reports/*.json 2>/dev/null || echo "No reports found"

report:
	./docker/run.sh python -c "from analysis import TrafficVisualizer, MetricsCalculator; print('Analysis tools ready')"

clean:
	docker compose down -v
	docker rm -f ai-traffic-testbed 2>/dev/null || true

clean-all: clean
	docker rmi 6g-ai-testbed:latest 2>/dev/null || true
	docker volume prune -f

# =============================================================================
# Development
# =============================================================================

lint:
	./docker/run.sh python -m flake8 --max-line-length=100 .

format:
	./docker/run.sh python -m black .

typecheck:
	./docker/run.sh python -m mypy .

# =============================================================================
# Environment Setup
# =============================================================================

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from template"; \
		echo "Please edit .env and add your API keys"; \
	else \
		echo ".env file already exists"; \
	fi

validate-env:
	@./docker/run.sh python -c "import os; print('OPENAI_API_KEY:', 'set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET')"
	@./docker/run.sh python -c "import os; print('BRAVE_API_KEY:', 'set' if os.environ.get('BRAVE_API_KEY') else 'NOT SET')"
